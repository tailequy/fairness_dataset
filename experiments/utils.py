import numpy as np
import sklearn.metrics as metrics
from scipy import interpolate
import matplotlib.pyplot as plt


def compute_roc(y_scores, y_true):
    """
    Function to compute the Receiver Operating Characteristic (ROC) curve for a set of predicted probabilities and the true class labels.
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns FPR and TPR values
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    return fpr, tpr


def compute_auc(y_scores, y_true):
    """
    Function to Area Under the Receiver Operating Characteristic Curve (AUC)
    y_scores - vector of predicted probability of being in the positive class P(X == 1) (numeric)
    y_true - vector of true labels (numeric)
    Returns AUC value
    """
    auc = metrics.roc_auc_score(y_true, y_scores)
    return auc


def interpolate_roc_fun(fpr, tpr, n_grid):
    """
    Function to Use interpolation to make approximate the Receiver Operating Characteristic (ROC) curve along n_grid equally-spaced values.
    fpr - vector of false positive rates computed from compute_roc
    tpr - vector of true positive rates computed from compute_roc
    n_grid - number of approximation points to use (default value of 10000 more than adequate for most applications) (numeric)

    Returns  a list with components x and y, containing n coordinates which  interpolate the given data points according to the method (and rule) desired
    """
    roc_approx = interpolate.interp1d(x=fpr, y=tpr)
    x_new = np.linspace(0, 1, num=n_grid)
    y_new = roc_approx(x_new)
    return x_new, y_new


def slice_plot(
    majority_roc_fpr,
    minority_roc_fpr,
    majority_roc_tpr,
    minority_roc_tpr,
    majority_group_name,
    minority_group_name,
    fout="./slice_plot.png",
    value=0.0
):
    """
    Function to create a 'slice plot' of two roc curves with area between them (the ABROCA region) shaded.

    majority_roc_fpr, minority_roc_fpr - FPR of majority and minority groups
    majority_roc_tpr, minority_roc_tpr - TPR of majority and minority groups
    majority_group_name - (optional) - majority group display name on the slice plot
    minority_group_name - (optional) - minority group display name on the slice plot
    fout - (optional) -  File name (including directory) to save the slice plot generated

    No return value; displays slice plot & file is saved to disk
    """
    plt.figure(1, figsize=(5, 4))
    title = "ABROCA = " + str(value)
    plt.title(title)
    plt.xlabel("False Positive Rate",fontweight='bold')
    plt.ylabel("True Positive Rate",fontweight='bold')
    plt.ylim((-0.04, 1.04))
    plt.plot(
        majority_roc_fpr,
        majority_roc_tpr,
        #label="{o} - Baseline".format(o=majority_group_name),
        label="{o}".format(o=majority_group_name),
        linestyle="-",
        color="r",
    )
    plt.plot(
        minority_roc_fpr,
        minority_roc_tpr,
        #label="{o} - Comparison".format(o=minority_group_name),
        label="{o}".format(o=minority_group_name),
        linestyle="-",
        color="b",
    )
    plt.fill(
        majority_roc_fpr.tolist() + np.flipud(minority_roc_fpr).tolist(),
        majority_roc_tpr.tolist() + np.flipud(minority_roc_tpr).tolist(),
        "y",
    )
    plt.legend()
    plt.savefig(fout,bbox_inches = "tight")
    plt.show()
