# Land Use and Land Cover Classification from Satellite Images using CNNs and Transfer Learning

Deep Learning course project – Group 7, DSEB65A, National Economics University (NEU).

We classify Sentinel-2 satellite images (EuroSAT RGB: 27,000 images, 64×64 px, 10 classes) with three models built in TensorFlow/Keras:

1. **Simple CNN** – convolution, pooling and fully connected layers
2. **Complex CNN** – 4 self-designed CNN blocks ([Conv 3×3 – ReLU] × 2 → MaxPool → Dropout), Global Average Pooling, fully connected layers; 3 variants compared (with/without BatchNorm, GAP vs Flatten)
3. **ResNet50V2** – transfer learning from ImageNet: frozen backbone first, then fine-tuning of the last stage (conv5)

## Results on the test set (4,050 images)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Parameters | Training time (min) |
|---|---|---|---|---|---|---|
| Simple CNN | 0.8328 | 0.8300 | 0.8269 | 0.8250 | 1,626,442 | 9.7 |
| Complex CNN | 0.9249 | 0.9271 | 0.9207 | 0.9227 | 1,240,618 | 85.4 |
| ResNet50V2 | 0.9595 | 0.9585 | 0.9580 | 0.9582 | 23,585,290 | 234.5 |

Source: `results/test_metrics.csv`. Precision, Recall and F1 are macro averages over the 10 classes. Training was done on CPU.

## Repository structure
```
├── src/
│   ├── config.py                  paths and settings shared by all scripts
│   ├── data_utils.py              load images, data augmentation
│   ├── train_utils.py             train + save a model, plot learning curves
│   ├── eval_utils.py              test metrics, confusion matrix, confused pairs
│   ├── 01_data_preprocessing.py   integrity check, EDA, 70/15/15 split, normalization, augmentation   (Req 1)
│   ├── 02_simple_cnn.py           Simple CNN                                                           (Req 2)
│   ├── 03_complex_cnn.py          Complex CNN with CNN blocks + 3 variants                             (Req 3)
│   ├── 04_transfer_learning.py    ResNet50V2: phase 1 (frozen) + phase 2 (fine-tune conv5)             (Req 4)
│   ├── 05_evaluation.py           test-set evaluation of the 3 models, comparison, error analysis      (Req 5)
│   └── 06_predict.py              load the saved models and predict (no training)
├── data/          splits.csv (the raw images are not on GitHub – see data/README.md)
├── results/       result tables, training histories, classification reports, run logs
├── figures/       figures used in the report
├── models/        trained models (created by the scripts, not on GitHub)
└── report/        final report
```

## How to run
**On your own computer** (Python 3.10+):
```bash
pip install -r requirements.txt
python src/01_data_preprocessing.py     # downloads the dataset if needed, creates data/splits.csv
python src/02_simple_cnn.py
python src/03_complex_cnn.py
python src/04_transfer_learning.py
python src/05_evaluation.py             # test-set evaluation of the 3 models
python src/06_predict.py                # load the saved models and predict the test set
```
Run the commands from the project folder, in this order. Each script prints its progress and saves its outputs to `models/`, `results/` and `figures/`.

Training takes several hours on CPU. Scripts 02-04 save each trained model to `models/` as a `.keras` file, and `06_predict.py` only loads these files and predicts, so it runs in a few minutes.

**On Google Colab** (faster with a GPU: `Runtime → Change runtime type → T4 GPU`):
```
!git clone <link to this repository>
%cd eurosat-lulc-classification
!python src/01_data_preprocessing.py
!python src/02_simple_cnn.py
!python src/03_complex_cnn.py
!python src/04_transfer_learning.py
!python src/05_evaluation.py
!python src/06_predict.py
```
Files created on Colab are deleted when the session ends – download `results/` and `figures/` (or copy them to Google Drive) before closing.

## Team
| Name | Role |
|---|---|
| | Role 1 – Data & preprocessing, Simple CNN |
| | Role 2 – Complex CNN, GitHub repository |
| | Role 3 – Transfer learning, evaluation & comparison |

## References
- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE JSTARS*, 12(7), 2217–2226.
- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). *ICLR*.
- Lin, M., Chen, Q., & Yan, S. (2014). Network in Network. *ICLR*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks (ResNetV2). *ECCV*.
