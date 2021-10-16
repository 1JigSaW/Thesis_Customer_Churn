import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore
from sklearn.metrics import (confusion_matrix,  # type: ignore
                             accuracy_score, 
                             roc_auc_score, 
                             plot_precision_recall_curve, 
                             plot_roc_curve, 
                             recall_score, 
                             precision_score,
                             f1_score)

knn = KNeighborsClassifier(n_jobs=-1)
gnb = GaussianNB()
svc = SVC(random_state=42, degree=3)
lgc = LogisticRegression(random_state=42)
dcc = DecisionTreeClassifier(random_state=42, max_depth=6)
rfc = RandomForestClassifier(random_state=42, min_samples_leaf=30)
lda = LinearDiscriminantAnalysis()

estimators = [('K-nearest neighbors', knn),
              ('Naive Bayes classifier', gnb),
              ('Support vector machine', svc),
              ('Logistic Regression', lgc), 
              ('Decision tree', dcc),
              ('Random forest', rfc),
              ('Linear discriminant analysis', lda)]

def confusion_plot(label: str,
                   y_valid: np.ndarray,
                   y_pred: np.ndarray,
                   ax: int = None):
    co_ma = confusion_matrix(y_valid, y_pred)
    groups = ['True Neg','False Pos','False Neg','True Pos']
    counts = [int(value) for value in co_ma.flatten()]
    shares = ['{0:.2%}'.format(value) for value in
              co_ma.flatten()/np.sum(co_ma)]
    labels_list = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                   zip(groups, counts, shares)]
    labels = np.asarray(labels_list).reshape(2,2)
    sns.heatmap(co_ma, annot=labels, cmap='binary', alpha=.55, ax=ax,
                cbar=True, fmt='', linewidth=1, linecolor='black')
    plt.title(f'Confusion Matrix for {label}')

def create_metrics(X_train: np.ndarray,
                   X_valid: np.ndarray,
                   y_train: np.ndarray,
                   y_valid: np.ndarray):
    metrics = []
    for est in estimators:

        fig, axes = plt.subplots(ncols=3, figsize=(15,4))

        mod = est[1].fit(X_train, y_train)
        y_pred = mod.predict(X_valid)
        plot_precision_recall_curve(mod, X_valid, y_valid, 
                                    y_pred, ax=axes[0], color='black')
        plot_roc_curve(mod, X_valid, y_valid, ax = axes[1], color='black')

        axes[0].set_title(f'Precision-Recall Curve for {est[0]}')
        axes[1].set_title(f'ROC Curve for {est[0]}')
        axes[1].plot([1,0], [1,0], c='green', ls='--')
        confusion_plot(est[0], y_valid, y_pred, axes[2])

        scores = {}
        scores['classifier'] = est[0]
        scores['accuracy_score'] = accuracy_score(y_valid, y_pred)
        scores['roc_auc_score'] = roc_auc_score(y_valid, y_pred)
        scores['f1_score'] = f1_score(y_valid, y_pred)
        scores['recall_score'] = recall_score(y_valid, y_pred)
        scores['precision_score'] = precision_score(y_valid, y_pred)

        plt.tight_layout()
        metrics.append(scores)
        plt.savefig('graph_' + str(est[0]))
    return pd.DataFrame(metrics)
