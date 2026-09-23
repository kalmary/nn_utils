from .src.accuracy_metrics import *
from .src.file_handling import *
from .src.loss_functions import *
from .src.training_callbacks import *


def __getattr__(name):
    if name in {"Plotter", "ClassificationReport"}:
        from .src import evaluation_plot_tools

        return getattr(evaluation_plot_tools, name)
    raise AttributeError(name)
