"""HW1 NumPy MLP package."""

from .data import SplitData, make_splits
from .evaluate import build_confusion_matrix, evaluate
from .model import MLPClassifier
from .search import run_search, search_space
from .trainer import TrainConfig, SGDOptimizer, train_model
from .visualize import (
    plot_confusion_matrix,
    plot_training_curves,
    save_error_analysis,
    save_weight_visualizations,
)
