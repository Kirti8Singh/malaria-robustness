import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import CLASS_TO_LABEL, IMAGE_SIZE, load_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "resnet18"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "sensitivity_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_images = 0
    all_labels = []
    all_probabilities = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probabilities.extend(
                probabilities.detach().cpu().numpy().tolist()
            )

    metrics = calculate_metrics(
        np.asarray(all_labels),
        np.asarray(all_probabilities),
    )
    metrics["loss"] = total_loss / total_images
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train an ImageNet-pretrained ResNet18 baseline."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)
    print("Seed:", args.seed)
    print("Batch size:", args.batch_size)
    print("Maximum epochs:", args.epochs)
    print("Early-stopping patience:", args.patience)
    print("Learning rate:", args.lr)

    _, datasets = load_datasets()

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("\nLoading ImageNet-pretrained ResNet18...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_LABEL))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Keep this run's artifacts separate from the preliminary one-epoch run.
    run_dir = OUTPUT_DIR / "final_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "resnet18_best.pt"
    history_path = run_dir / "training_history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, validation_loader, criterion, device
        )

        scheduler.step(val_metrics["loss"])

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"\nEpoch {epoch}/{args.epochs} | {time.time() - start_time:.1f}s"
            f"\n  Train: loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"F1={train_metrics['f1']:.4f}, "
            f"AUC={train_metrics['roc_auc']:.4f}"
            f"\n  Val:   loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "validation_metrics": val_metrics,
                    "class_to_label": CLASS_TO_LABEL,
                    "seed": args.seed,
                    "model_name": "resnet18",
                    "weights": "ImageNet DEFAULT",
                    "image_size": IMAGE_SIZE,
                    "normalization": "ImageNet mean/std",
                    "training_config": {
                        "max_epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                        "weight_decay": 1e-4,
                        "early_stopping_patience": args.patience,
                        "selection_metric": "validation_loss",
                    },
                },
                checkpoint_path,
            )
            print("  Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print(
                "  No validation-loss improvement: "
                f"{epochs_without_improvement}/{args.patience}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", f"{best_val_loss:.4f}")
    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
