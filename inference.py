"""
Simple inference script for Caelum Incognitum.
Usage: python inference.py --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


def load_model(model_path: str, config_path: str, device: torch.device):
    """Load trained model and config."""
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint["args"]
    
    # Determine number of classes
    if config["mode"] == "closed_set_4way":
        num_classes = 4
    else:
        num_classes = 3  # OSR mode
    
    # Build model
    backbone = args.get("backbone", "resnet18")
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model, config


def get_transform(img_size: int = 224):
    """Get inference transforms."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def confidence_score(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).max(dim=1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def predict(model, image_path: str, config: dict, device: torch.device):
    """Run prediction on a single image."""
    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    transform = get_transform()
    x = transform(img).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    # Get base prediction
    probs = F.softmax(logits, dim=1)
    base_pred_idx = logits.argmax(dim=1).item()
    base_confidence = probs[0, base_pred_idx].item()
    
    # Get all class probabilities
    known_classes = config.get("known_classes", {})
    
    if config["mode"] == "osr_threshold":
        # OSR mode - check threshold
        osr_method = config.get("osr_method", "confidence")
        threshold = config["threshold"]
        
        if osr_method == "confidence":
            score = confidence_score(logits).item()
        else:  # energy
            temperature = config.get("energy_temperature", 1.0)
            score = -energy_score(logits, temperature).item()  # negate so higher = more known
        
        if score < threshold:
            prediction = "unknown"
            is_unknown = True
        else:
            prediction = known_classes.get(str(base_pred_idx), f"class_{base_pred_idx}")
            is_unknown = False
        
        return {
            "prediction": prediction,
            "is_unknown": is_unknown,
            "confidence": base_confidence,
            "osr_score": score,
            "threshold": threshold,
            "osr_method": osr_method,
            "class_probabilities": {
                known_classes.get(str(i), f"class_{i}"): probs[0, i].item()
                for i in range(logits.shape[1])
            }
        }
    else:
        # Closed-set mode
        idx_to_class = config.get("idx_to_class", {})
        prediction = idx_to_class.get(str(base_pred_idx), f"class_{base_pred_idx}")
        
        return {
            "prediction": prediction,
            "confidence": base_confidence,
            "class_probabilities": {
                idx_to_class.get(str(i), f"class_{i}"): probs[0, i].item()
                for i in range(logits.shape[1])
            }
        }


def main():
    ap = argparse.ArgumentParser(description="Run inference on a single image")
    ap.add_argument("--image", type=str, required=True, help="Path to image file")
    ap.add_argument("--model", type=str, default="outputs/best_model15.pt", help="Path to model checkpoint")
    ap.add_argument("--config", type=str, default="outputs/config2.json", help="Path to config file")
    args = ap.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model, args.config, device)
    print(f"Loaded model (OSR method: {config.get('osr_method', 'n/a')})")
    
    # Run prediction
    result = predict(model, args.image, config, device)
    
    # Print results
    print("\n" + "="*50)
    print(f"Image: {args.image}")
    print("="*50)
    print(f"\n🎯 Prediction: {result['prediction'].upper()}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    
    if "is_unknown" in result:
        print(f"\n📊 OSR Details:")
        print(f"   Method: {result['osr_method']}")
        print(f"   Score: {result['osr_score']:.6f}")
        print(f"   Threshold: {result['threshold']:.6f}")
        print(f"   Status: {'REJECTED (unknown)' if result['is_unknown'] else 'ACCEPTED (known)'}")
    
    print(f"\n📈 Class Probabilities:")
    for cls, prob in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"   {cls:12} {prob*100:5.1f}% {bar}")


if __name__ == "__main__":
    main()