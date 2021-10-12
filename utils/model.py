import numpy as np
import matplotlib.pyplot as plt # type: ignore
from utils.plot import confusion_plot
from sklearn.metrics import (accuracy_score, # type: ignore
                             roc_auc_score,
                             plot_precision_recall_curve,
                             plot_roc_curve,
                             precision_score,
                             recall_score,
                             f1_score)
from sklearn.ensemble import BaseEnsemble # type: ignore

def gen_stacked_metrics(cls_name: str, y_valid: np.ndarray, y_pred: np.ndarray) -> dict:
    stacked_metrics = {}
    stacked_metrics['classifier'] = cls_name
    stacked_metrics['accuracy_score'] = accuracy_score(y_valid, y_pred)
    stacked_metrics['ROC AUC score'] = roc_auc_score(y_valid, y_pred)
    stacked_metrics['f1_score'] = f1_score(y_valid, y_pred)
    stacked_metrics['recall_score'] = recall_score(y_valid, y_pred)
    stacked_metrics['precision_score'] = precision_score(y_valid, y_pred)

    return stacked_metrics

def init_plot(cls_name: str, regmodel: BaseEnsemble, X_valid: np.ndarray, y_valid: np.ndarray, y_pred: np.ndarray):
    fig, axes = plt.subplots(ncols=3, figsize=(14,4))
    
    plot_precision_recall_curve(regmodel,X_valid,y_valid, 
                                y_pred, color='black', ax=axes[0])
    plot_roc_curve(regmodel,X_valid,y_valid, 
                   color='black', ax=axes[1])
    
    axes[0].set_title(f'Precision-Recall Curve for {cls_name}')
    axes[1].plot([1,0],[1,0], c='green',ls='--')
    axes[1].set_title(f'ROC Curve for {cls_name}')
    confusion_plot(cls_name,y_valid, y_pred, axes[2])
