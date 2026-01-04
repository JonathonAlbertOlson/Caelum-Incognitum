"""
Gradio web interface for Caelum Incognitum.
Run: python app.py
"""

import json
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# --- Load model once at startup ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Load config
with open("outputs/config.json", "r") as f:
    CONFIG = json.load(f)

# Load model
checkpoint = torch.load("outputs/best_model.pt", map_location=DEVICE)
args = checkpoint["args"]
num_classes = 3 if CONFIG["mode"] == "osr_threshold" else 4

if args.get("backbone", "resnet18") == "resnet18":
    MODEL = models.resnet18(weights=None)
    MODEL.fc = torch.nn.Linear(MODEL.fc.in_features, num_classes)
else:
    MODEL = models.resnet50(weights=None)
    MODEL.fc = torch.nn.Linear(MODEL.fc.in_features, num_classes)

MODEL.load_state_dict(checkpoint["model_state"])
MODEL.to(DEVICE)
MODEL.eval()
print("Model loaded!")

# Transform
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def confidence_score(logits):
    return F.softmax(logits, dim=1).max(dim=1).values


def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def predict(image):
    """Run prediction and return results."""
    if image is None:
        return "No image provided", "", {}
    
    # Preprocess
    img = Image.fromarray(image).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = MODEL(x)
    
    probs = F.softmax(logits, dim=1)
    pred_idx = logits.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    
    known_classes = CONFIG.get("known_classes", {})
    threshold = CONFIG.get("threshold", 0.5)
    osr_method = CONFIG.get("osr_method", "confidence")
    
    # Get OSR score
    if osr_method == "confidence":
        score = confidence_score(logits).item()
    else:
        temp = CONFIG.get("energy_temperature", 1.0)
        score = -energy_score(logits, temp).item()
    
    # Determine prediction
    if score < threshold:
        prediction = "❓ UNKNOWN"
        status = f"Rejected (score {score:.4f} < threshold {threshold:.4f})"
    else:
        class_name = known_classes.get(str(pred_idx), f"class_{pred_idx}")
        prediction = f"✅ {class_name.upper()}"
        status = f"Accepted (score {score:.4f} ≥ threshold {threshold:.4f})"
    
    # Class probabilities for label output
    class_probs = {
        known_classes.get(str(i), f"class_{i}"): float(probs[0, i])
        for i in range(logits.shape[1])
    }
    
    return prediction, status, class_probs


# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="OSR Status"),
        gr.Label(label="Class Probabilities", num_top_classes=3),
    ],
    title="🛸 Caelum Incognitum",
    description="UFO Classification with Open-Set Recognition. Upload an image of a flying object to classify it as aircraft, bird, drone, or unknown.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()