import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np
import seaborn as sns

import pathlib as pth
from typing import Union, Optional


class Plotter:
    """
    A utility class for generating various evaluation plots for machine learning models.
    
    Args:
        class_num (int): Number of classes in the classification problem
        plots_dir (Union[str, pth.Path]): Directory where plots will be saved
    """
    def __init__(self, class_num: int, plots_dir: Union[str, pth.Path]):
        self.class_num = class_num
        self.plots_dir = pth.Path(plots_dir)

    def plot_metric_hist(self,
                         file_name: str,
                         metric: list[float],
                         val_metric: Optional[list[float]] = None):
        """
        Plot training and validation metric history over epochs.
        
        Args:
            file_name (str): Name of the output file
            metric (list[float]): Training metric values per epoch
            val_metric (Optional[list[float]]): Validation metric values per epoch
        """
        file_path = self.plots_dir.joinpath(file_name)
        metric_name = file_name.split('_')[0]

        plt.figure(figsize=(10, 5))
        plt.plot(metric)
        
        if val_metric is not None:
            plt.plot(val_metric)
        plt.xlabel('Epoch [n]')
        plt.ylabel(f'{metric_name} [-]')
        plt.title(f'{metric_name} progression during training')
        plt.tight_layout()

        legend_lst = [f'{metric_name}']
        if val_metric is not None:
            legend_lst.append(f'{metric_name} - validation')

        plt.legend(legend_lst)

        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

    def cnf_matrix(self, file_name: str,
                        target: np.ndarray, 
                        prediction: np.ndarray,
                        num_classes: Optional[int] = None,
                        class_names: Optional[list] = None):  
            
        """
        Plots a confusion matrix heatmap showing the relationship between actual and predicted classes.

        Args:
            file_name (str): Name of the output file where the plot will be saved
            target (np.ndarray): Ground truth labels
            prediction (np.ndarray): Model predictions 
            num_classes (Optional[int]): Number of classes. If None, inferred from confusion matrix shape
            class_names (Optional[list]): List of class names to use as labels. If None, default names are generated

        The plot shows:
        - Heatmap with color intensity indicating number of samples
        - Numeric values in each cell showing the raw counts
        - Class labels on both axes
        - Overall accuracy percentage in the title
        """
                   
        file_path = self.plots_dir.joinpath(file_name)

        cm = confusion_matrix(y_true=target, y_pred=prediction)
        accuracy = accuracy_score(y_true=target, y_pred=prediction)
        if class_names is None:
            n_classes = num_classes if num_classes is not None else cm.shape[0]
            class_names = [f"Class_{i}" for i in range(n_classes)]

        plt.figure(figsize=(8, 6)) 
        annot_kws = {"fontweight": 'bold'}
        sns.heatmap(cm, 
                    annot=True, 
                    annot_kws=annot_kws,      
                    fmt='d',
                    cmap='Purples',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar=True,
                    linewidths=0.4,
                    linecolor='black') 

        plt.title(f"Correlation matrix\nAccuracy: {accuracy:.2%}")
        plt.ylabel("Actual classes")
        plt.xlabel("Estimated classes")

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def cnf_matrix_analysis(self, file_name: str,
                target: np.ndarray, 
                prediction: np.ndarray,
                num_classes: Optional[int] = None,
                class_names: Optional[list] = None):
        """
            Plots an enhanced confusion matrix heatmap with additional percentage analysis.

            Args:
                file_name (str): Name of the output file where the plot will be saved
                target (np.ndarray): Ground truth labels
                prediction (np.ndarray): Model predictions
                num_classes (Optional[int]): Number of classes. If None, inferred from confusion matrix shape
                class_names (Optional[list]): List of class names to use as labels. If None, default names are generated

            The plot shows:
            - Heatmap with color intensity indicating number of samples
            - Both raw counts and percentages in each cell
            - Empty cells for zero counts
            - Class labels on both axes
            - Overall accuracy percentage in the title
            - Enhanced visual formatting including:
        """         


        file_path = self.plots_dir.joinpath(file_name)

        cm = confusion_matrix(y_true=target, y_pred=prediction)
        accuracy = accuracy_score(y_true=target, y_pred=prediction)

        if class_names is None:
            n_classes = num_classes if num_classes is not None else cm.shape[0]
            class_names = [f"Class_{i}" for i in range(n_classes)]

        row_sums = cm.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_perc = cm / row_sums[:, np.newaxis]
        cm_perc = np.nan_to_num(cm_perc)

        annot_labels = np.empty_like(cm).astype(object)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                count = cm[i, j]
                percent = cm_perc[i, j]
                
                if count == 0:
                    annot_labels[i, j] = ""
                else:

                    annot_labels[i, j] = f"{count}\n({percent:.1%})"

        sns.set_context("notebook", font_scale=1.1)
        plt.figure(figsize=(10, 8))


        ax = sns.heatmap(cm, 
                        annot=annot_labels, 
                        fmt='', 
                        cmap='Purples', 
                        xticklabels=class_names, 
                        yticklabels=class_names,
                        cbar_kws={'label': 'Liczba próbek'},
                        linewidths=0.8,
                        linecolor='black') 


        plt.title(f"Correlation matrix\n Accuracy: {accuracy:.2%}", fontsize=14, pad=20, weight='bold')
        plt.ylabel("Actual classes", fontsize=12, labelpad=10)
        plt.xlabel("Estimated classes", fontsize=12, labelpad=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    def prc_curve(self, file_name: str,
                  target: np.ndarray, pred_prob: np.ndarray):
        """
        Plot Precision-Recall curves for multi-class classification.
        
        Args:
            file_name (str): Name of the output file
            target (np.ndarray): Ground truth labels
            pred_prob (np.ndarray): Predicted probabilities for each class
        """
        file_path = self.plots_dir.joinpath(file_name)
        legend = []

        plt.figure(figsize=(8, 6))

        for i in range(self.class_num):

            legend.append(f'Curve for: Class_{i}')
            precision, recall, thresholds = precision_recall_curve(target == i, pred_prob[:, i])
            plt.plot(recall, precision, label=legend[i])

        plt.xlabel('Recall')
        plt.legend()
        plt.title(f'Curve precision and recall')
        plt.ylabel('Precision')
        plt.grid(True)

        plt.savefig(file_path)
        plt.close()

    def roc_curve(self, file_name: str,
                  target: np.ndarray, pred_prob: np.ndarray):
        """
        Plot ROC (Receiver Operating Characteristic) curves for multi-class classification.
        
        Args:
            file_name (str): Name of the output file
            target (np.ndarray): Ground truth labels
            pred_prob (np.ndarray): Predicted probabilities for each class
        """
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
                         pred: np.ndarray, target: np.ndarray,
                         additional_info: Optional[str] = None) -> None:
    """
    Generate and save a detailed classification report to a text file.
    
    Args:
        file_path (Union[str, pth.Path]): Path where the report will be saved
        pred (np.ndarray): Model predictions
        target (np.ndarray): Ground truth labels
    
    The report includes precision, recall, f1-score, and support for each class.
    """
    file_path = pth.Path(file_path)

    pred = pred.flatten()
    target = target.flatten()

    report: str = classification_report(target, pred)

    if additional_info is not None:
        report += f'\nAdditional info: {additional_info}'

    with open(file_path, 'w') as f:
        f.write(report)
