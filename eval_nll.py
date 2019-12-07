import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def load_avg_nlls(nll_dir, dataset):
    return np.load(nll_dir + '/' + dataset + '_avg_graph_nlls.npy')

def plot_nlls(nlls, title, xlabel):
    """
        Plot the nll distribution for a set of graphs

        Parameters:
        - nlls: list of nlls calculates
    """
    fig, ax = plt.subplots()
    # Plot the two distributions side by side
    sns.distplot(nlls, ax=ax, kde=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    
    return fig, ax


def compare_dist(nlls_list, labels, title):
    """
        Plot multiple graph distributions

        Parameters:
        - nlls_list: list of nll arrays for the different graph classes
        we are comparing
        - lables: the labels that will be used in the legend
            - Examples: DD_1 train (normal), DD_1 (normal), DD_2 (anom)
    """
    fig, ax = plt.subplots()
    for i in range(len(nlls_list)):
        sns.distplot(nlls_list[i], ax=ax, kde=True, label=labels[i])
    ax.legend()
    ax.set_xlabel("Negative Log Likelihood")
    ax.set_title(title)
    return fig, ax

def anomally_detection_score(nlls, labels, threshold):
    """
        Given nll predictions for a set of graphs, classify
        the graphs into nomral and anomalous graph classes
        based on the nll threshold
    """
    pred_labels = np.zeros(nlls.shape[0])

    for i in range(nlls):
        # Label 1 is anomalous
        if nlls[i] > threshold:
            pred_labels[0] = 1

    # Compute the accuracy
    accuracy = np.sum(np.abs(pred_labels - labels)) / labels.shape[0]
    return accuracy

    