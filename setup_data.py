"""
setup_data.py — Run this ONCE after downloading the dataset.
Saves class names to data/class_names.json so the API can use them.
"""
import os
import json
import argparse
from torchvision import datasets


def setup(train_dir="data/plantvillage/train"):
    if not os.path.exists(train_dir):
        print(f"Dataset not found at: {train_dir}")
        print("\nDownload it first:")
        print("  kaggle datasets download -d vipoooool/new-plant-diseases-dataset "
              "-p data/plantvillage --unzip")
        return

    ds = datasets.ImageFolder(train_dir)
    print(f"Found {len(ds.classes)} classes, {len(ds)} total images")
    print(f"First 5 classes: {ds.classes[:5]}")

    os.makedirs("data", exist_ok=True)
    out = "data/class_names.json"
    with open(out, "w") as f:
        json.dump(ds.classes, f, indent=2)
    print(f"\nClass names saved → {out}")

    # also update config with actual num_classes
    import yaml
    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["num_classes"] = len(ds.classes)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Updated num_classes={len(ds.classes)} in {cfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="data/plantvillage/train")
    args = parser.parse_args()
    setup(args.train_dir)
