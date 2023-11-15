import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve


def plot_roc_curve(predicted_prob, y_label, linestyles, labels, file_name=None):
    """
    Plot the roc curves for multiple models
    Args:
        predicted_prob: list of arrays, each array contains de probability of an event to be a goal, predicted by the models
        y_test: list of arrays, each array contains the actual label of an event
        linestyles: list of strings, the linestyle to use for each curve
        labels: list of strings, labels to give in the legend
        file_name: str, if None, do nothing, if it's a string, save the figure name as the string.
        
    Returns:
    
    """
    for prob, y, linestyle, label in zip(predicted_prob, y_label, linestyles, labels):
        roc_auc = roc_auc_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, linestyle=linestyle, label=label + f' (area={roc_auc:.2f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves and the AUC metric of the ROC Curve')
    plt.legend()
    plt.grid(True)

    if file_name:
        plt.savefig(file_name)
    plt.show()


def shot_prob_model_percentile(pred_probs, y_label):
    """
    Create the shot probability model percentile
    Args:
        pred_probs: array of probabilities
        y_label: actual label of each event

    Returns: the array of percentile, the array of probabilities that correspond to each percentile and
    a dataframe saying to which bin of percentile an event corresponds

    """
    percentile = np.arange(0, 102, 2)
    pred_percentile = np.percentile(pred_probs, percentile)
    pred_percentile = np.concatenate([[0], pred_percentile])
    pred_percentile = np.unique(pred_percentile)

    y_val_df = pd.DataFrame(y_label)
    y_val_df.rename(columns={y_val_df.columns[0]: "isGoal"}, inplace=True)
    y_val_df['percentile_bin'] = pd.cut(pred_probs, pred_percentile, include_lowest=True)

    return percentile, pred_percentile, y_val_df


def plot_goal_rate(pred_probs, y_label, labels, file_name=None):
    """
    Create the goal rate plot
    Args:
        pred_probs: list of arrays, each array contains de probability of an event to be a goal
        y_label: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        file_name: str, if None, do nothing, if it's a string, save the figure name as the string.

    Returns:

    """
    for prob, y, label in zip(pred_probs, y_label, labels):
        percentile, percentile_pred, y_val_df = shot_prob_model_percentile(prob, y)
        bins = np.linspace(0, 100, len(y_val_df['percentile_bin'].unique()))[1:]

        goal_rate_by_percentile_bin = y_val_df.groupby(by=['percentile_bin']).apply(
            lambda f: f['isGoal'].sum() / len(f))

        g = sns.lineplot(x=bins, y=goal_rate_by_percentile_bin[1:] * 100, label=label)
        ax = g.axes
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('shot probability model percentile')
    plt.ylabel('#Goals / (#no_Goals + #Goals)')
    plt.title('Goal rate as a function of shot probability model percentile')
    plt.grid(True)

    if file_name:
        plt.savefig(file_name)
    plt.show()

def plot_goal_rate_5models(pred_probs, y_label, labels, file_name=None):
    """
    Create the goal rate plot
    Args:
        pred_probs: list of arrays, each array contains de probability of an event to be a goal
        y_label: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        file_name: str, if None, do nothing, if it's a string, save the figure name as the string.

    Returns:

    """
    for prob, y, label in zip(pred_probs, y_label, labels):
        percentile, percentile_pred, y_val_df = shot_prob_model_percentile(prob, y)
        bins = np.linspace(0, 100, len(y_val_df['percentile_bin'].unique()))[1:]

        goal_rate_by_percentile_bin = y_val_df.groupby(by=['percentile_bin']).apply(lambda f: f['isGoal'].sum()/(len(f)+1))

        # print("Length of goal_rate_by_percentile_bin:", len(goal_rate_by_percentile_bin))
        # print(len(bins))
        # print("Length of index:", len(goal_rate_by_percentile_bin.index))
        # print(goal_rate_by_percentile_bin)
        # goal_rate_by_percentile_bin = y_val_df.groupby(by=['percentile_bin'])['isGoal'].mean()
        if label != 'XGBoost':
            g = sns.lineplot(x=bins, y=goal_rate_by_percentile_bin[1:]*100, label=label)
            ax = g.axes
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
        else:
            bins = np.linspace(0, 100, len(y_val_df['percentile_bin'].unique())+1)[1:]
            g = sns.lineplot(x=bins, y=goal_rate_by_percentile_bin[1:]*100, label=label)
            ax = g.axes
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('shot probability model percentile')
    plt.ylabel('#Goals / (#no_Goals + #Goals)')
    plt.title('Goal rate of shot probability model percentile')
    plt.grid(True)
    
    if file_name:
        plt.savefig(file_name)
    plt.show()
    
def plot_cumulative_sum(pred_probs, y_label, labels, file_name=None):
    """
    Create cumulative sum of goals plot
    Args:
        pred_probs: list of arrays, each array contains de probability of an event to be a goal
        y_label: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        file_name: str, if None, do nothing, if it's a string, save the figure name as the string.

    Returns:

    """
    for prob, y, label in zip(pred_probs, y_label, labels):
        percentile, percentile_pred, y_val_df = shot_prob_model_percentile(prob, y)
        number_goal_sum = (y == 1).sum()
        sum_goals_by_percentile = y_val_df.groupby(by='percentile_bin').apply(
            lambda f: f['isGoal'].sum() / number_goal_sum)
        cumu_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]
        bins = np.linspace(0, 100, len(y_val_df['percentile_bin'].unique()))[1:]

        g = sns.lineplot(x=bins, y=cumu_sum_goals[1:] * 100, label=label)
        ax = g.axes
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Cumulative sum')
    plt.title('Cumulative proportion of goals of shot probability model percentile')

    if file_name:
        plt.savefig(file_name)
    plt.show()


def plot_calibration(pred_probs, y_label, labels, file_name=None):
    """
    Create calibration plot
    Args:
        pred_probs: list of arrays, each array contains de probability of an event to be a goal
        y_label: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        file_name: str, if None, do nothing, if it's a string, save the figure name as the string.

    Returns:

    """
    # sns.set_theme()
    fig = plt.figure()
    ax = plt.axes()
    for prob, y, label in zip(pred_probs, y_label, labels):
        disp = CalibrationDisplay.from_predictions(y, prob, n_bins=25, ax=ax, name=label, ref_line=True)
    plt.xlim(0, 1)
    plt.legend(loc=9)
    plt.title('Calibration curve')

    if file_name:
        plt.savefig(file_name)
    plt.show()


def plot_roc_curve_nn(predictions, y_true):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, predictions)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def shot_prob_model_percentile_nn(predictions, y_true):
    label = ['isGoal']
    percentile = np.arange(0, 102, 2)
    pred_percentile = np.percentile(predictions, percentile)
    pred_percentile = np.concatenate([[0], pred_percentile])
    pred_percentile = np.unique(pred_percentile)

    y1_val_df = pd.DataFrame(y_true)
    y1_val_df.rename(columns={y1_val_df.columns[0]: "isGoal"}, inplace=True)
    y1_val_df['percentile_bin'] = pd.cut(predictions, pred_percentile, include_lowest=True)

    bins = np.linspace(0, 100, len(y1_val_df['percentile_bin'].unique()))[1:]

    goal_rate_by_percentile_bin = y1_val_df.groupby(by=['percentile_bin']).apply(lambda f: f['isGoal'].sum() / len(f))

    g = sns.lineplot(x=bins, y=goal_rate_by_percentile_bin[1:] * 100, label=label)

    ax = g.axes
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('shot probability model percentile')
    plt.ylabel('#Goals / (#no_Goals + #Goals)')
    plt.title('Goal rate as a function of shot probability model percentile')
    plt.grid(True)
    plt.show()


def plot_cumulative_sum_nn(predictions, y_true):
    label = ['isGoal']

    percentile = np.arange(0, 102, 2)
    pred_percentile = np.percentile(predictions, percentile)
    pred_percentile = np.concatenate([[0], pred_percentile])
    pred_percentile = np.unique(pred_percentile)

    y1_val_df = pd.DataFrame(y_true)
    y1_val_df.rename(columns={y1_val_df.columns[0]: "isGoal"}, inplace=True)
    y1_val_df['percentile_bin'] = pd.cut(predictions, pred_percentile, include_lowest=True)

    number_goal_sum = (y_true == 1).sum()
    sum_goals_by_percentile = y1_val_df.groupby(by='percentile_bin').apply(
        lambda f: f['isGoal'].sum() / number_goal_sum)
    cumu_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]
    bins = np.linspace(0, 100, len(y1_val_df['percentile_bin'].unique()))[1:]

    g = sns.lineplot(x=bins, y=cumu_sum_goals[1:] * 100, label=label)
    ax = g.axes
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))

    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Cumulative sum')
    plt.title('Cumulative proportion of goals as a function of shot probability model percentile')
    plt.show()


def plot_calibration_curve_nn(predictions, y_true):
    # reliability diagram
    fop, mpv = calibration_curve(y_true, predictions, n_bins=10)

    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')

    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.title('Calibration curve')
    plt.show()
