"""
Download diverse images for the 'unknown' class from Tiny ImageNet.
This gives your OSR model exposure to varied real-world images.

Usage: python download_unknowns.py --output data/unknown --count 500
"""

import argparse
import random
from pathlib import Path

def download_with_huggingface(output_dir: Path, count: int):
    """Download random images from Tiny ImageNet via HuggingFace."""
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Install required packages:")
        print("  pip install datasets pillow")
        return
    
    print("Loading Tiny ImageNet dataset...")
    ds = load_dataset('Maysee/tiny-imagenet', split='train')
    
    # Get random indices
    indices = random.sample(range(len(ds)), min(count, len(ds)))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(indices)} random images to {output_dir}...")
    for i, idx in enumerate(indices):
        img = ds[idx]['image']
        # Resize to match your training size
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(output_dir / f"unknown_{i:05d}.jpg")
        
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(indices)} images")
    
    print(f"Done! Saved {len(indices)} images to {output_dir}")


def download_with_torchvision(output_dir: Path, count: int):
    """Alternative: Download using torchvision (CIFAR-100 for diversity)."""
    try:
        from torchvision import datasets
        from PIL import Image
    except ImportError:
        print("Install required packages:")
        print("  pip install torchvision pillow")
        return
    
    print("Downloading CIFAR-100 dataset...")
    ds = datasets.CIFAR100(root='./cifar100_temp', train=True, download=True)
    
    indices = random.sample(range(len(ds)), min(count, len(ds)))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(indices)} random images to {output_dir}...")
    for i, idx in enumerate(indices):
        img, _ = ds[idx]
        # CIFAR is 32x32, resize to 224
        img = img.resize((224, 224), Image.LANCZOS)
        img.save(output_dir / f"unknown_{i:05d}.jpg")
        
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(indices)} images")
    
    print(f"Done! Saved {len(indices)} images to {output_dir}")
    print("You can delete ./cifar100_temp folder if you want")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="data/unknown", 
                    help="Output directory for unknown images")
    ap.add_argument("--count", type=int, default=500,
                    help="Number of images to download")
    ap.add_argument("--source", choices=["tiny-imagenet", "cifar100"], 
                    default="tiny-imagenet",
                    help="Dataset source")
    args = ap.parse_args()
    
    output_dir = Path(args.output)
    
    # Clear existing balloon-only images (optional - comment out if you want to keep them)
    # if output_dir.exists():
    #     print(f"Warning: {output_dir} already exists. New images will be added.")
    
    if args.source == "tiny-imagenet":
        download_with_huggingface(output_dir, args.count)
    else:
        download_with_torchvision(output_dir, args.count)
    
    print("\nNext steps:")
    print("1. Optionally keep some of your original balloon images")
    print("2. Retrain: python train_cnn_autosplit.py --data_root data --pretrained")


if __name__ == "__main__":
    main()