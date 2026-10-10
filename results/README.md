# Results

All files are created by the scripts in `src/`.

| File | Content | Script |
|---|---|---|
| `data_check.csv` | Integrity check: corrupt / wrong size / not RGB / duplicate images | 01 |
| `split_summary.csv` | Images per class in train / val / test | 01 |
| `*_history.csv` | Loss and accuracy per epoch (train and validation) | 02, 03, 04 |
| `*_summary.json` | Parameters, epochs run, best epoch, best val_loss / val_accuracy, training time | 02, 03, 04 |
| `complex_cnn_variants.csv` | Comparison of the 3 Complex CNN variants | 03 |
| `test_metrics.csv` | **Main table:** test Accuracy, macro Precision / Recall / F1, parameters, training time | 05 |
| `per_class_f1.csv` | F1-score of each class for the 3 models | 05 |
| `top_confused_pairs.csv` | Most confused class pairs of the best model | 05 |
| `classification_reports/` | Full per-class report of each model | 05 |
| `logs/` | Console output of each script from the run that produced these results | 01–05 |
