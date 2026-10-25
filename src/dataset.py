from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "results/dataset/master_metadata_patient_split_balanced.csv"
)

IMAGE_SIZE = 224

# ImageNet normalization is appropriate for an ImageNet-pretrained ResNet18.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Explicit mapping: parasitized is the positive class.
CLASS_TO_LABEL = {
    "uninfected": 0,
    "parasitized": 1,
}


def get_transforms():
    """Return deterministic preprocessing for the baseline experiment."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MalariaDataset(Dataset):
    def __init__(self, metadata, split, transform=None):
        self.metadata = metadata.loc[
            metadata["split"] == split
        ].reset_index(drop=True)

        self.transform = transform

        if self.metadata.empty:
            raise ValueError(f"No images found for split: {split}")

        if self.metadata["relative_path"].isna().any():
            raise ValueError(f"Missing image paths in {split} split")

        unknown_classes = (
            set(self.metadata["class"].unique())
            - set(CLASS_TO_LABEL)
        )
        if unknown_classes:
            raise ValueError(f"Unknown classes: {unknown_classes}")

        self.paths = [
            PROJECT_ROOT / path
            for path in self.metadata["relative_path"]
        ]
        self.labels = [
            CLASS_TO_LABEL[label]
            for label in self.metadata["class"]
        ]

        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image files are missing. "
                f"First missing path: {missing[0]}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        label = self.labels[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_datasets():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {
        "filename", "patient_id", "class", "split", "relative_path"
    }
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError("Required metadata fields contain missing values")

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(metadata["split"].unique())
    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected split names: {actual_splits}"
        )

    # Confirm that no patient appears in more than one split.
    patient_split_counts = metadata.groupby("patient_id")["split"].nunique()
    if (patient_split_counts > 1).any():
        raise ValueError("Patient leakage detected across splits")

    # Each relative path should identify one image record only.
    if metadata["relative_path"].duplicated().any():
        raise ValueError("Duplicate image paths found in metadata")

    transform = get_transforms()

    datasets = {
        split: MalariaDataset(metadata, split, transform)
        for split in ("train", "validation", "test")
    }

    return metadata, datasets


if __name__ == "__main__":
    metadata, datasets = load_datasets()

    print("Dataset verification passed.")
    print(f"Total metadata rows: {len(metadata)}")
    print(
        "Unique patients:",
        metadata["patient_id"].nunique()
    )

    for split, dataset in datasets.items():
        subset = metadata[metadata["split"] == split]
        print(f"\n{split.upper()}")
        print(f"Images: {len(dataset)}")
        print(f"Patients: {subset['patient_id'].nunique()}")
        print("Class counts:")
        print(subset["class"].value_counts().to_string())

    print("\nClass mapping:", CLASS_TO_LABEL)
