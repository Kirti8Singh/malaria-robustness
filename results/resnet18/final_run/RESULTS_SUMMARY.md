# ResNet18 Baseline Results

## Configuration
- Architecture: ResNet18
- Initial weights: ImageNet pretrained
- Input size: 224 x 224
- Batch size: 16
- Learning rate: 0.0001
- Random seed: 42
- Checkpoint selection: Lowest validation loss
- Best epoch: 4
- Test set: Fixed patient-level held-out split
- Number of test images: 4137

## Clean-Test Metrics
- Accuracy: 0.970510
- Precision: 0.976051
- Sensitivity / Recall: 0.964734
- Specificity: 0.976294
- F1-score: 0.970360
- ROC-AUC: 0.992957

## Confusion Matrix
| Actual class | Predicted uninfected | Predicted parasitized |
|---|---:|---:|
| Uninfected | 2018 | 49 |
| Parasitized | 73 | 1997 |

These results describe clean-image performance only. Robustness under image-quality degradations will be evaluated separately.
