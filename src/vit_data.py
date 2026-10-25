"""Member-2-only dataset adapter for the repository's audited fixed split.

Reads results/dataset/master_metadata_patient_split_balanced.csv and image
files from cell_images/. It never creates or modifies any split or metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_INDEX = {"uninfected": 0, "parasitized": 1}
SPLITS = ("train", "validation", "test")


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


class MalariaImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, project_root: Path,
                 transform: Callable | None = None):
        self.rows = rows.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform or build_transform()
        self.paths = [self._resolve_path(str(value)) for value in self.rows["relative_path"]]
        self.labels = [CLASS_TO_INDEX[str(value).strip().lower()] for value in self.rows["class"]]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            examples = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} image(s) referenced by the fixed metadata are missing. "
                f"First missing paths:\n{examples}\nExpected dataset at {project_root / 'cell_images'}"
            )

    def _resolve_path(self, relative_path: str) -> Path:
        # Metadata paths are repository-relative, e.g. cell_images/Parasitized/....png
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"Expected a repository-relative image path, got: {relative_path}")
        resolved = (self.project_root / path).resolve()
        root = self.project_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Image path escapes project root: {relative_path}")
        return resolved

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, self.labels[index]


def load_vit_datasets(project_root: str | Path = ".", metadata_path: str | Path | None = None):
    """Return (metadata_frame, {train, validation, test: Dataset})."""
    root = Path(project_root).expanduser().resolve()
    csv_path = Path(metadata_path).expanduser() if metadata_path else (
        root / "results" / "dataset" / "master_metadata_patient_split_balanced.csv"
    )
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Fixed master metadata not found: {csv_path}. Obtain the audited metadata file; "
            "do not regenerate or alter the split for the ViT experiment."
        )

    metadata = pd.read_csv(csv_path)
    required = {"filename", "patient_id", "class", "split", "relative_path"}
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if metadata[list(required)].isnull().any().any():
        raise ValueError("Metadata contains null values in required columns.")
    metadata["class"] = metadata["class"].astype(str).str.strip().str.lower()
    metadata["split"] = metadata["split"].astype(str).str.strip().str.lower()
    unknown_classes = sorted(set(metadata["class"]) - set(CLASS_TO_INDEX))
    unknown_splits = sorted(set(metadata["split"]) - set(SPLITS))
    if unknown_classes or unknown_splits:
        raise ValueError(f"Unexpected class values {unknown_classes} or split values {unknown_splits}")

    # Fail closed if the fixed metadata contains patient leakage; never repair/re-split it.
    split_counts = metadata.groupby("patient_id")["split"].nunique()
    leaking_patients = split_counts[split_counts > 1].index.tolist()
    if leaking_patients:
        raise ValueError(f"Patient leakage in existing metadata; refusing to train. Examples: {leaking_patients[:5]}")

    datasets = {
        split: MalariaImageDataset(metadata.loc[metadata["split"] == split], root)
        for split in SPLITS
    }
    if any(len(datasets[split]) == 0 for split in SPLITS):
        raise ValueError(f"At least one fixed split is empty: { {k: len(v) for k, v in datasets.items()} }")
    return metadata, datasets
