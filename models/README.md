# Trained models

The `.keras` files are too large for GitHub (the fine-tuned ResNet50V2 alone is over 200 MB), so they are blocked by `.gitignore`. Running the scripts creates them here.

| File | Model | Created by |
|---|---|---|
| `simple_cnn.keras` | Simple CNN | `src/02_simple_cnn.py` |
| `complex_cnn_v1.keras`, `complex_cnn_no_bn.keras`, `complex_cnn_flatten.keras` | The 3 Complex CNN variants | `src/03_complex_cnn.py` |
| `complex_cnn_best.keras` | Copy of the best Complex CNN variant (lowest val_loss) | `src/03_complex_cnn.py` |
| `resnet50v2_phase1.keras` | ResNet50V2 after phase 1 (frozen backbone) | `src/04_transfer_learning.py` |
| `resnet50v2_finetuned.keras` | ResNet50V2 after fine-tuning conv5 | `src/04_transfer_learning.py` |

`src/05_evaluation.py` and `src/06_predict.py` need `simple_cnn.keras`, `complex_cnn_best.keras` and `resnet50v2_finetuned.keras`.
The models were saved with TensorFlow 2.21 / Keras 3.15. Loading them needs Keras 3 (TensorFlow 2.16 or newer).
