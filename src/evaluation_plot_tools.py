import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np


import pathlib as pth
from typing import Union, Optional




class Plotter:
    def __init__(self, class_num: int, plots_dir: Union[str, pth.Path]):
        self.class_num = class_num
        self.plots_dir = pth.Path(plots_dir)
        assert self.plots_dir.is_dir(), f'Plot directory {self.plots_dir} is not a directory'

        if not self.plot_dir.exists():
            self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_loss(self, file_name: str,
                  loss: list[float], val_loss: Optional[list[float]] = None):

        file_path = self.plots_dir.joinpath(file_name)
        plt.figure(figsize=(10, 5))
        plt.plot(loss)
        if val_loss is not None:
            plt.plot(val_loss)
        plt.xlabel('Epoch [n]')
        plt.ylabel('Loss [-]')
        plt.title('Loss history during training.')
        plt.tight_layout()
        if val_loss is not None:
            plt.legend(['loss', 'loss - validation'])
        else:
            plt.legend(['loss'])
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def plot_accuracy(self, file_name: str,
                      accuracy: list[float], val_accuracy: Optional[list[float]] = None):

        file_path = self.plots_dir.joinpath(file_name)
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy)
        if val_accuracy is not None:
            plt.plot(val_accuracy)
        plt.xlabel('Epoch [n]')
        plt.ylabel('Acuracy [-]')
        plt.title('Accuracy history during training.')
        plt.tight_layout()
        if val_accuracy is not None:
            plt.legend(['accuracy', 'accuracy - validation'])
        else:
            plt.legend(['accuracy'])
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def plot_miou(self, file_name: str,
                      miou: list[float], val_miou: Optional[list[float]] = None):

        file_path = self.plots_dir.joinpath(file_name)
        plt.figure(figsize=(10, 5))
        plt.plot(miou)
        if val_miou is not None:
            plt.plot(val_miou)
        plt.xlabel('Epoch [n]')
        plt.ylabel('mIoU [-]')
        plt.title('mIoU history during training')
        plt.tight_layout()
        if val_miou is not None:
            plt.legend(['mIoU', 'mIoU - validation'])
        else:
            plt.legend(['mIoU'])
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()


    def cnf_matrix(self, file_name: str,
                   target: np.ndarray, prediction: np.ndarray):
        file_path = self.plots_dir.joinpath(file_name)

        accuracy = accuracy_score(y_true=target, y_pred=prediction)
        cm = confusion_matrix(y_true=target, y_pred=prediction)

        # Print accuracy
        print(f"Accuracy: {accuracy:.4f}")

        # Visualize confusion matrix using heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.title("Confusion matrix")
        plt.ylabel("Target classes")
        plt.xlabel("Estimated classes")

        class_names = [f"Class_{i}" for i in range(self.class_num)]
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.mean() else 'black')

        plt.tight_layout()

        plt.savefig(file_path)
        plt.close()

    def prc_curve(self, file_name: str,
                  target: np.ndarray, pred_prob: np.ndarray):


        file_path = self.plots_dir.joinpath(file_name)
        legend = []

        plt.figure(figsize=(8, 6))

        for i in range(self.class_num):

            legend.append(f'Curve of Class_{i}')
            precision, recall, thresholds = precision_recall_curve(target == i, pred_prob[:, i])
            plt.plot(recall, precision, label=legend[i])

        plt.xlabel('Recall')
        plt.legend()
        plt.title(f'Precision recall curve')
        plt.ylabel('Precision')
        plt.grid(True)

        plt.savefig(file_path)
        plt.close()

    def roc_curve(self, file_name: str,
                  target: np.ndarray, pred_prob: np.ndarray):

        file_path = self.plots_dir.joinpath(file_name)

        y_true_bin = label_binarize(target, classes=np.arange(self.class_num))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.class_num):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(self.class_num):
            plt.plot(fpr[i], tpr[i], label=f'curve for area_{i}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve')
        plt.legend(loc="lower right")


        plt.savefig(file_path)
        plt.close()

def ClassificationReport(file_path: Union[str, pth.Path],
                         pred: np.ndarray, target: np.ndarray) -> None:

    file_path = pth.Path(file_path)

    pred = pred.flatten()
    target = target.flatten()

    report: str = classification_report(target, pred)

    with open(file_path, 'w') as f:
        f.write(report)
