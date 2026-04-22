# HW1: EuroSAT Classification with a NumPy MLP

This project implements a three-layer MLP from scratch with `NumPy` for EuroSAT RGB image classification.

The code includes:

- manual forward propagation
- manual backward propagation
- SGD training
- learning-rate decay
- L2 regularization
- hyperparameter search
- test evaluation and confusion matrix

## Environment

- Python 3.9+
- numpy
- Pillow
- matplotlib
- scikit-learn
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset in the project root like this:

```text
hw1/
├─ EuroSAT_RGB/
├─ src/
├─ batch_test_trials.py
├─ error_analysis.py
├─ train.py
├─ run_search.py
├─ test.py
├─ visualize_weights.py
└─ requirements.txt
```

## Train

### Run hyperparameter search

```bash
python run_search.py --data-root EuroSAT_RGB --output-dir outputs --epochs 20 --batch-size 128
```

### Run a single training job

```bash
python train.py --data-root EuroSAT_RGB --output-dir outputs/single --hidden-dim1 1024 --hidden-dim2 512 --activation relu --learning-rate 0.03 --weight-decay 0.001 --epochs 20 --batch-size 128
```

## Test

Test a saved checkpoint:

```bash
python test.py --data-root EuroSAT_RGB --weights path/to/best_model.npz --output-dir path/to/output_dir
```

`test.py` will automatically read `train_config.json` from the same directory as the checkpoint.

## Additional Utilities

### Batch test all trials

Evaluate all `trial_*` folders under one output directory and summarize their test performance:

```bash
python batch_test_trials.py --data-root EuroSAT_RGB --trials-root path/to/trials_root --summary-dir path/to/summary_dir
```

This script generates merged summary tables and comparison figures for all trials.

### Visualize first-layer weights

Visualize the first hidden-layer weights of a trained model:

```bash
python visualize_weights.py --data-root EuroSAT_RGB --weights path/to/best_model.npz --output-dir path/to/output_dir
```

This script saves a visualization image of the first-layer weights.

### Export error analysis results

Export misclassified test examples and common confusion pairs:

```bash
python error_analysis.py --data-root EuroSAT_RGB --weights path/to/best_model.npz --output-dir path/to/output_dir --max-examples 12
```

This script saves wrong-case visualization images and an error summary file.

## Main Files

- `train.py`: single-run training
- `run_search.py`: hyperparameter search
- `test.py`: test evaluation
- `batch_test_trials.py`: batch evaluation and summary of all trials
- `visualize_weights.py`: first-layer weight visualization
- `error_analysis.py`: export error-analysis materials
- `src/hw1_mlp/model.py`: MLP model and manual backpropagation
- `src/hw1_mlp/data.py`: data loading and preprocessing
