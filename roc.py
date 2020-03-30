import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt


def plot_roc(positive_data, negative_data, col="b", label=None):
    y_true = np.concatenate([np.zeros(np.shape(negative_data)[0]), np.ones(np.shape(positive_data)[0])])
    y_score = np.concatenate([negative_data, positive_data])
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)

    print('AUPR (%):', round(100 * sklearn.metrics.average_precision_score(y_true, y_score), 2))
    print('AUROC (%):', round(100 * sklearn.metrics.roc_auc_score(y_true, y_score), 2))

    plt.plot(fpr, tpr, color=col, lw=2, alpha=.8, label=label)
    return fpr, tpr, thresholds


def pr_curve(true_data, false_data, col="b", label=None):
    y_true = np.concatenate([np.zeros(np.shape(false_data)[0]), np.ones(np.shape(true_data)[0])])
    y_score = np.concatenate([false_data, true_data])
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, color=col)
