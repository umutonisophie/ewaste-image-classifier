"""Load the trained e-waste classifier and predict a device from an image.

The checkpoint shipped in this project would not load: its zip archive had a
loose `split_dataset.py` at the root, which torch refuses to read. Removing
that one entry produces a checkpoint that loads cleanly and matches a 29-class
MobileNetV3-Large exactly - see `repair_checkpoint.py`.

Usage:
    python predict.py path/to/image.jpg
    python predict.py --dir MASTER_e_waste_visual_dataset     # evaluate a folder
    python predict.py --checkpoint ewaste_mobilenet_master_model.repaired.pth IMAGE
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "ewaste_mobilenet_master_model.repaired.pth"
LABELS_FILE = HERE / "labels.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels(path: Path = LABELS_FILE) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_model(checkpoint: Path):
    """Recreate the 29-class MobileNetV3-Large and load the trained weights.

    torchvision's constructor defaults to 1000 ImageNet classes, so num_classes
    has to be set explicitly or the classifier head will not fit.
    """
    labels = load_labels()
    model = models.mobilenet_v3_large(weights=None, num_classes=len(labels))

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"Checkpoint does not fit the model: {len(missing)} missing keys, "
            f"{len(unexpected)} unexpected keys."
        )
    model.eval()
    return model, labels


def preprocessor() -> transforms.Compose:
    """ImageNet-style preprocessing.

    The classifier head was retrained but the backbone was not, so the input
    scaling the backbone expects still applies.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model, labels, transform, image_path: Path, top_k: int = 3):
    tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    probs = model(tensor).softmax(dim=1)[0]
    scores, indices = probs.topk(min(top_k, len(labels)))
    return [(labels[i], float(s)) for s, i in zip(scores, indices)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="path to a single image")
    parser.add_argument("--dir", help="evaluate every image in a folder tree")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"error: checkpoint not found at {checkpoint}", file=sys.stderr)
        print("Run repair_checkpoint.py first, or train a new model.", file=sys.stderr)
        return 1

    model, labels = build_model(checkpoint)
    transform = preprocessor()

    if args.dir:
        root = Path(args.dir)
        images = sorted(p for p in root.rglob("*")
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"error: no images found under {root}", file=sys.stderr)
            return 1

        correct = 0
        scored = 0
        for image in images:
            # An image directly inside a class folder carries its label from the
            # folder name; images elsewhere are predicted but not scored.
            truth = image.parent.name
            ranked = predict(model, labels, transform, image, args.top_k)
            has_truth = truth in labels
            if has_truth:
                scored += 1
                if ranked[0][0] == truth:
                    correct += 1
            best = ", ".join(f"{n} {c:.2f}" for n, c in ranked)
            mark = ""
            if has_truth:
                mark = "  OK" if ranked[0][0] == truth else "  MISS"
            print(f"{image.name:28} -> {best}{mark}")

        if scored:
            print(f"\ntop-1 accuracy: {correct}/{scored} ({correct/scored:.1%})")
        else:
            print("\nno images sat in a labelled class folder, so nothing was scored")
    elif args.image:
        image = Path(args.image)
        if not image.exists():
            print(f"error: image not found at {image}", file=sys.stderr)
            return 1
        for name, score in predict(model, labels, transform, image, args.top_k):
            print(f"{name:28} {score:.4f}")
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
