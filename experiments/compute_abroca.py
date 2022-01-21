from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing


def compute_abroca(
    df,
    pred_col,
    label_col,
    protected_attr_col,
    majority_protected_attr_val,
    n_grid=10000,
    plot_slices=False,
    lb=0,
    ub=1,
    limit=1000,
    file_name="slice_image.png",
    majority_group_name = 'Male',
    minority_group_name = 'Female'
):
    # Compute the value of the abroca statistic.
    """
    df - dataframe containing colnames matching pred_col, label_col and protected_attr_col
    pred_col - name of column containing predicted probabilities (string)
    label_col - name of column containing true labels (should be 0,1 only) (string)
    protected_attr_col - name of column containing protected attribute (should be binary) (string)
    majority_protected_attr_val name of 'majority' group with respect to protected attribute (string)
    n_grid (optional) - number of grid points to use in approximation (numeric) (default of 10000 is more than adequate for most cases)
    plot_slices (optional) - if TRUE, ROC slice plots are generated and saved to file_name (boolean)
    lb (optional) - Lower limit of integration (use -numpy.inf for -infinity) Default is 0
    ub (optional) - Upper limit of integration (use -numpy.inf for -infinity) Default is 1
    limit (optional) - An upper bound on the number of subintervals used in the adaptive algorithm.Default is 1000
    file_name (optional) - File name (including directory) to save the slice plot generated

    Returns Abroca value
    """
    if df[pred_col].between(0, 1, inclusive=True).any():
        pass
    else:
        print("predictions must be in range [0,1]")
        exit(1)
    if len(df[label_col].value_counts()) == 2:
        pass
    else:
        print("The label column should be binary")
        exit(1)
    if len(df[protected_attr_col].value_counts()) == 2:
        pass
    else:
        print("The protected attribute column should be binary")
        exit(1)
    # initialize data structures
    # slice_score = 0
    prot_attr_values = df[protected_attr_col].value_counts().index.values
    fpr_tpr_dict = {}

    # compute roc within each group of pa_values
    for pa_value in prot_attr_values:
        if pa_value != majority_protected_attr_val:
            minority_protected_attr_val = pa_value
        pa_df = df[df[protected_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col], pa_df[label_col])

    # compare minority to majority class; accumulate absolute difference btw ROC curves to slicing statistic
    majority_roc_x, majority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[majority_protected_attr_val][0],
        fpr_tpr_dict[majority_protected_attr_val][1],
        n_grid,
    )
    minority_roc_x, minority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[minority_protected_attr_val][0],
        fpr_tpr_dict[minority_protected_attr_val][1],
        n_grid,
    )

    # use function approximation to compute slice statistic via piecewise linear function
    if list(majority_roc_x) == list(minority_roc_x):
        f1 = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - minority_roc_y))
        f2 = lambda x, acc: abs(f1(x))
        slice, _ = integrate.quad(f2, lb, ub, limit)
    else:
        print("Majority and minority FPR are different")
        exit(1)

    if plot_slices == True:
        slice_plot(
            majority_roc_x,
            minority_roc_x,
            majority_roc_y,
            minority_roc_y,
            majority_group_name=majority_group_name,
            minority_group_name=minority_group_name,
            fout=file_name,
            value=round(slice,4),
        )

    return slice
