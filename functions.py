import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import confusion_matrix, auc, roc_curve


def distribution_plot(df, col="age"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(data=df, x=col, fill=True, ax=axs[0])
    axs[0].set_title(f"Overall {format_feature_name(col)} Distribution")
    sns.kdeplot(data=df, x=col, hue="stroke", common_norm=False, fill=True, ax=axs[1])
    axs[1].set_title(f"{format_feature_name(col)}-Stroke Distribution")
    axs[1].legend(title="", loc="upper right", labels=["Had a stroke", "Healthy"])


def format_feature_name(name):
    words = name.split("_")
    formatted_name = " ".join(word.lower().capitalize() for word in words)
    return formatted_name


def normalized_barplots(
    df,
    cols,
    hue,
    grid_x,
    grid_y,
    palette,
    figsize=(12, 8),
    legend_loc=(0.2, 1.5),
    **kwargs,
):
    num_plots = len(cols)

    fig, axs = plt.subplots(grid_y, grid_x, figsize=figsize, sharey=True)

    legend_created = False

    if grid_x > 1 and grid_y > 1:
        axs = axs.reshape(-1)

    for i, col in enumerate(cols):
        if grid_x > 1 or grid_y > 1:
            row = i // grid_x
            col = i % grid_x if grid_x > 1 else i // grid_y
        else:
            col = i
        df1 = df.groupby(cols[i])[hue].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename("percent").reset_index()
        sns.barplot(
            x=cols[i], y="percent", hue=hue, data=df1, ax=axs[i], palette=palette
        )
        sns.despine()
        axs[i].set_title(f"{format_feature_name(cols[i])} and risk of {hue}")
        axs[i].set_xlabel(f"{format_feature_name(cols[i])}")
        axs[i].get_legend().set_visible(False)

        if not legend_created:
            axs[i].legend(bbox_to_anchor=legend_loc, loc="upper right", borderaxespad=0)
            legend_created = True

        for label in axs[i].containers:
            axs[i].bar_label(label, fmt="%.2f%%")

    for i in range(num_plots, grid_y * grid_x):
        if grid_x > 1 or grid_y > 1:
            row = i // grid_x
            col = i % grid_x if grid_x > 1 else i // grid_y
        else:
            col = i
        axs[i].axis("off")

    plt.tight_layout()


def corr_heatmap(df, columns=None, method="pearson", figsize=(10, 5), **kwargs):
    if columns is not None:
        df = df[columns]

    if method in ["pearson", "kendall", "spearman"]:
        corr = df.corr(method=method)
        vmin = -1
        vmax = 1
        center = None
    elif method == "chi_squared":
        cat_var_combinations = list(itertools.combinations(columns, 2))
        result = {}
        for combination in cat_var_combinations:
            cross_tab = pd.crosstab(df[combination[0]], df[combination[1]])
            chi2, p, _, _ = ss.chi2_contingency(cross_tab)
            result[(combination[0], combination[1])] = p
        corr = pd.DataFrame(index=columns, columns=columns, dtype=float)
        np.fill_diagonal(corr.values, 1.0)
        for combination in cat_var_combinations:
            corr.at[combination[0], combination[1]] = result[
                (combination[0], combination[1])
            ]
            corr.at[combination[1], combination[0]] = result[
                (combination[0], combination[1])
            ]
        vmin = 0
        vmax = 1
        center = 0.05
    else:
        raise ValueError(
            "Invalid method specified. Valid options are 'pearson', 'kendall', 'spearman', or 'chi_squared'."
        )
    plt.figure(figsize=figsize)

    plt.title(f"{format_feature_name(method)} Correlation Matrix")

    mask = np.triu(np.ones_like(corr))

    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        center=center,
        **kwargs,
    )


def plot_confusion_matrix(
    cm,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=False,
    sum_stats=True,
    cmap="Blues",
    title="Confusion Matrix",
    figsize=None,
    ax=None,
) -> None:
    """
    Plot a heatmap representation of the confusion matrix along with optional summary statistics.

    Parameters:
        cm (numpy.ndarray): The confusion matrix to be visualized. It should be a 2D array-like object.
        group_names (list, optional): A list of strings representing the labels for each class. Default is None.
        categories ({"auto", list}, optional): The categories to be displayed on the x and y-axis.
            If "auto", the categories will be inferred from `group_names`. If a list is provided,
            it should match the length of `cm`. Default is "auto".
        count (bool, optional): If True, display the count of occurrences in each cell of the matrix.
            Default is True.
        percent (bool, optional): If True, display the percentage of occurrences in each cell of the matrix.
            Default is True.
        cbar (bool, optional): If True, display a colorbar alongside the heatmap. Default is False.
        sum_stats (bool, optional): If True, calculate and display summary statistics such as accuracy,
            precision, recall, and F1-score (applicable for binary confusion matrices). Default is True.
        cmap (str, optional): The color map to be used for the heatmap. Default is "Blues".
        figsize (tuple, optional): The size of the figure. If not provided, the default size will be used.

    Returns:
        None: This function only displays the heatmap plot and optional summary statistics.

    Note:
        - If `group_names` is provided, it must match the length of the confusion matrix.
        - The summary statistics are shown only if `sum_stats` is True and the confusion matrix is binary
          (i.e., has only two classes).

    Examples:
        # Example 1: Plot a confusion matrix with default settings
        plot_confusion_matrix(cm=[[10, 2], [3, 15]])

        # Example 2: Plot a confusion matrix with custom labels and no summary statistics
        plot_confusion_matrix(cm=[[5, 1, 2], [0, 7, 0], [3, 0, 8]], group_names=['A', 'B', 'C'], sum_stats=False)
    """
    blanks = ["" for i in range(cm.size)]

    if group_names and len(group_names) == cm.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    else:
        group_counts = blanks

    if percent:
        negative_percentages = [
            "{0:.2%}".format(value) for value in cm[0, :].flatten() / np.sum(cm[0, :])
        ]
        positive_percentages = [
            "{0:.2%}".format(value) for value in cm[1, :].flatten() / np.sum(cm[1, :])
        ]

        group_percentages = list(negative_percentages + positive_percentages)
    else:
        group_percentages = blanks

    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(cm.shape[0], cm.shape[1])
    if sum_stats:
        accuracy = np.trace(cm) / float(np.sum(cm))

        if len(cm) == 2:
            precision = cm[1, 1] / sum(cm[:, 1])
            recall = cm[1, 1] / sum(cm[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        square=True,
        xticklabels=categories,
        yticklabels=categories,
    )
    ax.set(
        title=title,
        xlabel="Predicted label" + stats_text,
        ylabel="Actual label",
    )
    if ax is None:
        plt.show()


def plot_roc_curve(
    fpr, tpr, fpr_2=None, tpr_2=None, label=None, label_2=None, figsize=(8, 6)
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for one or two sets of data.

    Parameters:
    fpr (array-like): An array of false positive rates.
    tpr (array-like): An array of true positive rates.
    fpr_2 (array-like, optional): An array of false positive rates for a second dataset. Default is None.
    tpr_2 (array-like, optional): An array of true positive rates for a second dataset. Default is None.
    label (str, optional): Label for the main dataset's ROC curve. Default is None.
    label_2 (str, optional): Label for the second dataset's ROC curve. Default is None.
    figsize (tuple, optional): A tuple specifying the figure size. Default is (8, 6).

    Returns:
    None

    This function creates a ROC curve plot with optional support for a second dataset's curve.
    If a second dataset is provided, its ROC curve will be plotted with a dashed blue line.
    The diagonal line representing random guessing is also plotted as a black dashed line.
    The plot includes labels, title, and grid for better visualization.

    If label_2 is provided, a legend will be displayed in the lower right corner of the plot.

    Example:
    >>> fpr = [0.0, 0.1, 0.2, 0.3, 0.4]
    >>> tpr = [0.2, 0.4, 0.6, 0.8, 1.0]
    >>> plot_roc_curve(fpr, tpr, label="Dataset 1")
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    if fpr_2 is not None:
        plt.plot(fpr_2, tpr_2, "b:", label=label_2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (Fall-Out)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.grid(True)
    if label_2 is not None:
        plt.legend(loc="lower right")
    plt.show()


class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.

    Parameters
    ----------
    group_cols : list
        List of columns used for calculating the aggregated value
    target : str
        The name of the column to impute
    metric : str
        The metric to be used for replacement, can be one of ['mean', 'median']

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    """

    def __init__(self, group_cols, target, metric="mean"):
        assert metric in [
            "mean",
            "median",
        ], "Unrecognized value for metric, should be mean/median"
        assert type(group_cols) == list, "group_cols should be a list of columns"
        assert type(target) == str, "target should be a string"

        self.group_cols = group_cols
        self.target = target
        self.metric = metric

    def fit(self, X, y=None):
        assert (
            pd.isnull(X[self.group_cols]).any(axis=None) == False
        ), "There are missing values in group_cols"

        impute_map = (
            X.groupby(self.group_cols)[self.target]
            .agg(self.metric)
            .reset_index(drop=False)
        )

        self.impute_map_ = impute_map

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "impute_map_")

        X = X.copy()

        for index, row in self.impute_map_.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.target] = X.loc[ind, self.target].fillna(row[self.target])

        return X


def binary_performances(
    y_true, y_prob, thresh=0.5, labels=["Positives", "Negatives"], categories=[0, 1]
):
    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError("A binary class problem is required")
        else:
            y_prob = y_prob[:, 1]

    plt.figure(figsize=[15, 4])

    # 1 -- Confusion matrix
    cm = confusion_matrix(y_true, (y_prob > thresh).astype(int))

    ax = plt.subplot(131)
    plot_confusion_matrix(cm, categories=categories, sum_stats=False, ax=ax)

    # 2 -- Distributions of Predicted Probabilities of both classes
    plt.subplot(132)
    plt.hist(
        y_prob[y_true == 1],
        density=True,
        bins=25,
        alpha=0.5,
        color="green",
        label=labels[0],
    )
    plt.hist(
        y_prob[y_true == 0],
        density=True,
        bins=25,
        alpha=0.5,
        color="red",
        label=labels[1],
    )
    plt.axvline(thresh, color="blue", linestyle="--", label="Boundary")
    plt.xlim([0, 1])
    plt.title("Distributions of Predictions", size=13)
    plt.xlabel("Positive Probability (predicted)", size=10)
    plt.ylabel("Samples (normalized scale)", size=10)
    plt.legend(loc="upper right")

    # 3 -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)
    plt.subplot(133)
    plt.plot(
        fp_rates,
        tp_rates,
        color="orange",
        lw=1,
        label="ROC curve (area = %0.3f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp / (fp + tn), tp / (tp + fn), "bo", markersize=8, label="Decision Point")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", size=10)
    plt.ylabel("True Positive Rate", size=10)
    plt.title("ROC Curve", size=13)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=0.3)
    plt.show()

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    results = {"Precision": precision, "Recall": recall, "F1 Score": F1, "AUC": roc_auc}

    prints = [f"{kpi}: {round(score, 3)}" for kpi, score in results.items()]
    prints = " | ".join(prints)
    print(prints)

    return results
