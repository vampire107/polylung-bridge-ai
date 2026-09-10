import argparse
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import Swin_T_Weights, swin_t


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def compute_binary_class_weights(dataset, class_names: List[str], device: str) -> torch.Tensor:
    # ImageFolder keeps integer labels in dataset.targets.
    labels = [class_names[idx] for idx in dataset.targets]
    binary = [0 if name.lower() == "normal" else 1 for name in labels]

    normal_count = sum(1 for x in binary if x == 0)
    pathological_count = sum(1 for x in binary if x == 1)
    if normal_count == 0 or pathological_count == 0:
        raise ValueError("Both binary classes must be present in training data for class-balanced loss")

    total = normal_count + pathological_count
    # Inverse-frequency weighting, normalized to keep average weight near 1.
    normal_w = total / (2.0 * normal_count)
    pathological_w = total / (2.0 * pathological_count)
    return torch.tensor([normal_w, pathological_w], dtype=torch.float32, device=device)


def to_binary_targets(targets: torch.Tensor, class_names: List[str]) -> torch.Tensor:
    labels = [class_names[idx] for idx in targets.tolist()]
    # Normal stays 0; all pathology classes map to 1.
    binary = [0 if name.lower() == "normal" else 1 for name in labels]
    return torch.tensor(binary, dtype=torch.long, device=targets.device)


def train_one_epoch(model, loader, optimizer, criterion, class_names, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        binary_targets = to_binary_targets(targets, class_names)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, binary_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, class_names, device) -> Dict:
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            binary_targets = to_binary_targets(targets, class_names)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            labels.extend(binary_targets.cpu().tolist())

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, target_names=["normal", "pathological"], output_dict=True)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "binary_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": {
            "labels": ["normal", "pathological"],
            "matrix": cm,
        },
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PD-1: Fine-tune Swin-T on LungHist700 binary objective")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/val folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="evidence/public/pd1/pd1_metrics.json")
    parser.add_argument("--published-baseline", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument(
        "--disable-class-balanced-loss",
        action="store_true",
        help="Disable inverse-frequency class weights in CrossEntropyLoss",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise FileNotFoundError("Expected train/ and val/ under --data-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size, args.num_workers)
    class_names = train_loader.dataset.classes

    weights = None
    if args.pretrained:
        try:
            weights = Swin_T_Weights.IMAGENET1K_V1
        except Exception:
            weights = None

    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.disable_class_balanced_loss:
        class_weights = None
    else:
        class_weights = compute_binary_class_weights(train_loader.dataset, class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    best_macro_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, class_names, device)
        eval_metrics = evaluate(model, val_loader, class_names, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 4),
                "val_binary_accuracy": eval_metrics["binary_accuracy"],
                "val_macro_f1": eval_metrics["macro_f1"],
                "val_weighted_f1": eval_metrics["weighted_f1"],
            }
        )
        if eval_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = eval_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate(model, val_loader, class_names, device)
    baseline_delta = round(final["binary_accuracy"] - args.published_baseline, 4)
    output = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "class_balanced_loss": not args.disable_class_balanced_loss,
        "binary_accuracy": final["binary_accuracy"],
        "macro_f1": final["macro_f1"],
        "weighted_f1": final["weighted_f1"],
        "published_baseline": args.published_baseline,
        "baseline_delta": baseline_delta,
        "history": history,
        "confusion_matrix": final["confusion_matrix"],
        "classification_report": final["classification_report"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote PD-1 metrics to {out_path}")

    model_output = Path(
        "evidence/public/pd1/pd1_swin_l40s_model.pt"
    )
    model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": train_loader.dataset.classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        model_output,
    )

    print(
        f"Saved model checkpoint to: {model_output}"
    )


if __name__ == "__main__":
    main()


