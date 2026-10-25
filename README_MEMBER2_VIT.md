# Member 2 — ViT-B/16 clean baseline

This is the Member 2 implementation only. It does not implement CNN–ViT fusion, degradation generation, or robustness experiments, and it never writes over ResNet18 artifacts.

## Repository integration

Copy `src/vit_data.py`, `src/train_vit.py`, and `src/evaluate_vit.py` into the repository's `src/` directory. The currently visible GitHub repository has the fixed split and metadata-generation script, but the `src/dataset.py` mentioned in the project brief was not present in the inspected branch. Therefore, `vit_data.py` is a **model-specific adapter** that reads the already-created `results/dataset/master_metadata_patient_split_balanced.csv`; it does not regenerate or edit the split and does not modify the shared ResNet pipeline. If your local `src/dataset.py` exists and is the agreed interface, the adapter can later be wired to it after checking its implementation.

Expected project layout:

```text
malaria-robustness/
├── cell_images/Parasitized/ and cell_images/Uninfected/
├── results/dataset/master_metadata_patient_split_balanced.csv
└── src/
    ├── vit_data.py
    ├── train_vit.py
    └── evaluate_vit.py
```

The dataset itself is not stored in this package. Keep the audited metadata and image directory in the project root. `vit_data.py` fails loudly if metadata/images are missing or patient leakage is detected; it never attempts to repair the split.

## Environment and smoke test

Activate the existing virtual environment and check the installed PyTorch/torchvision versions and device. Install compatible versions of `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, and `scikit-learn` if needed. The pretrained torchvision weights are downloaded on first use.

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
python -m src.train_vit --epochs 1 --batch-size 2 --max-train-batches 2 --output-dir results/vit_smoke
```

The smoke test verifies wiring only; it is not a final result. With ViT-B/16, use MPS when available on Apple Silicon, CUDA when available, otherwise CPU. Reduce batch size if memory is insufficient.

## Full run and test evaluation

```bash
python -m src.train_vit --epochs 10 --batch-size 8 --lr 1e-5 --weight-decay 1e-4 --seed 42 --output-dir results/vit
python -m src.evaluate_vit --checkpoint results/vit/vit_best.pt --output-dir results/vit --batch-size 8
```

The default settings are starting values, not claimed optimal hyperparameters. Select checkpoints and tune settings using validation only. Run test evaluation once the training configuration is finalized.

Outputs in `results/vit/`: `config.json`, `training_history.csv`, `training_summary.json`, `vit_best.pt`, `test_metrics.json`, `test_confusion_matrix.csv`, and `test_predictions.csv`.

Metrics include accuracy, precision, sensitivity/recall, specificity, F1 and ROC-AUC. Class mapping is uninfected=0 and parasitized=1. The best checkpoint is selected by minimum validation loss. The fixed patient-level split is never changed.

## Branch workflow

```bash
git status
git fetch origin
git branch -a
```

Switch to the existing `member2-vit` branch if present; otherwise create it from the latest `main` only after confirming the working tree is clean. Commit only the Member 2 files and do not commit the dataset or overwrite `results/resnet18/`.

## Checkpoint loading

```python
import torch
from src.train_vit import build_model
model = build_model(pretrained=False)
checkpoint = torch.load("results/vit/vit_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The responsibility is complete only after the full training run and held-out test evaluation have succeeded and their actual metrics are recorded. This package has not been trained against the private/local dataset yet, so no performance results are claimed.

## Verification performed in the preparation environment

- Python syntax compilation passed for all three source modules.
- A synthetic-data adapter test passed: it verified the three split loaders, RGB conversion, 224×224 normalization pipeline, and class mapping.
- A one-epoch CPU smoke test with randomly initialized weights passed, including checkpoint saving.
- Evaluation against that synthetic fixture passed and wrote metrics, confusion matrix, and per-image predictions.

These tests verify code wiring only. They do **not** represent model performance on the real malaria dataset. The full baseline still requires the real local image dataset and audited metadata, ImageNet weights, and a complete training/evaluation run.
