import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn

from src.dataset import CLASS_TO_LABEL, load_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "results" / "resnet18" / "final_run"
CHECKPOINT_PATH = RUN_DIR / "resnet18_best.pt"
METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset"
    / "master_metadata_patient_split_balanced.csv"
)


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_LABEL))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, datasets = load_datasets()
    test_dataset = datasets["test"]
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    labels_all = []
    predictions_all = []
    probabilities_all = []

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test images:", len(test_dataset))
    print("Evaluating...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= 0.5).long()

            labels_all.extend(labels.numpy().tolist())
            predictions_all.extend(predictions.cpu().numpy().tolist())
            probabilities_all.extend(probabilities.cpu().numpy().tolist())

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(predictions_all)
    y_prob = np.asarray(probabilities_all)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "ResNet18 ImageNet-pretrained",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "class_to_label": CLASS_TO_LABEL,
    }

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata.loc[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    if len(test_metadata) != len(y_true):
        raise ValueError(
            f"Test metadata has {len(test_metadata)} rows but "
            f"{len(y_true)} predictions were generated."
        )

    # Verify the loader's label order matches the metadata's label order.
    expected_labels = test_metadata["class"].map(CLASS_TO_LABEL).to_numpy()
    if not np.array_equal(expected_labels, y_true):
        raise ValueError(
            "Test labels and metadata order do not match. "
            "Predictions were not saved."
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RUN_DIR / "test_metrics.json"
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    predictions_path = RUN_DIR / "test_predictions.csv"
    output = test_metadata.copy()
    output["true_label"] = y_true
    output["predicted_label"] = y_pred
    output["parasitized_probability"] = y_prob
    output.to_csv(predictions_path, index=False)

    print("\nFINAL CLEAN-TEST RESULTS")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nMetrics saved to:", metrics_path)
    print("Predictions saved to:", predictions_path)


if __name__ == "__main__":
    main()
