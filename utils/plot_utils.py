import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, roc_curve



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
        plt.plot(fpr, tpr, linestyle=linestyle, label=label+f' (area={roc_auc:.2f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
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
    y_val_df.rename(columns={ y_val_df.columns[0]: "isGoal" }, inplace = True)
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

        goal_rate_by_percentile_bin = y_val_df.groupby(by=['percentile_bin']).apply(lambda f: f['isGoal'].sum()/len(f))

        g = sns.lineplot(x=bins, y=goal_rate_by_percentile_bin[1:]*100, label=label)
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
        number_goal_sum = (y==1).sum()
        sum_goals_by_percentile = y_val_df.groupby(by='percentile_bin').apply(lambda f: f['isGoal'].sum()/number_goal_sum)
        cumu_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]
        bins = np.linspace(0, 100, len(y_val_df['percentile_bin'].unique()))[1:]

        g = sns.lineplot(x=bins, y=cumu_sum_goals[1:]*100, label=label)
        ax = g.axes
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
        
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Cumulative sum')
    plt.title('Cumulative proportion of goals as a function of shot probability model percentile')
    
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
        disp = CalibrationDisplay.from_predictions(y, prob, n_bins=25, ax=ax, name=label, ref_line=False)
    plt.xlim(0, 1)
    plt.legend(loc=9)
    plt.title('Calibration curve')
    
    if file_name:
        plt.savefig(file_name)
    plt.show()