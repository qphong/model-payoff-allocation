import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import seaborn as sns

colors = sns.color_palette("tab10")

#####
# I/O
def get_collaboration_information(info_path):
    with open("{}.pkl".format(info_path), "rb") as infile:
        info = pickle.load(infile)

    value_func = lambda c: info["test_acc"][c] # including duplicated parties
    party_labels = [str(l) for l in info["party_labels"]] # including duplicated parties
    replicated_party_idxs = info["replicated_party_idxs"]
    n_party = len(info["party_labels"]) # including duplicated parties

    # what if we want to duplicate parties by code instead of training?
    # --> then we need a function to convert the non-duplicated info to duplicated info!

    return n_party, value_func, party_labels, replicated_party_idxs



###########################
# Handle replicated parties

def combine_replicate_payoff(payoff_mat, replicated_party_idxs):
    delete_idxs = []
    for idxs in replicated_party_idxs:
        for i in idxs[1:]:
            payoff_mat[idxs[0],:] += payoff_mat[i,:]
            payoff_mat[:,idxs[0]] += payoff_mat[:,i]
            delete_idxs.append(i)
    
    increase_delete_idxs = np.sort(delete_idxs)
    for i in increase_delete_idxs[::-1]:
        payoff_mat = np.delete(payoff_mat, i, axis=0)
        payoff_mat = np.delete(payoff_mat, i, axis=1)

    return payoff_mat

def remove_replicate_party_labels(party_labels, replicated_party_idxs):
    for idxs in replicated_party_idxs:
        for i in np.sort(idxs[1:])[::-1]:
            del party_labels[i] # delete in decreasing order of index

    return party_labels
    
def remove_replicate_party_self_values(self_values, replicated_party_idxs):
    for idxs in replicated_party_idxs:
        for i in np.sort(idxs[1:])[::-1]:
            self_values = np.delete(self_values, i)

    return self_values


###############
# Visualization

def visualize_payoff(payoff_mat, party_labels, saved_fn_prefix = None):
    """
    payoff_mat:
        row: outgoing (axis = 1)
        column: incoming (axis = 0)
        model_reward = total outgoing + self value
        income = total incoming - total outgoing
    """
    n_parties = payoff_mat.shape[0]

    # visualize payoff matrix
    fig, ax = plt.subplots(figsize=(5,5))
    # the below does not work for seaborn=0.9.0
    # im = ax.imshow(payoff_mat, cmap=sns.color_palette("dark:salmon_r", as_cmap=True))
    im = ax.imshow(payoff_mat, cmap=ListedColormap(sns.color_palette("dark:salmon_r")))
     
    ax.set_xticks(np.arange(n_parties))
    ax.set_yticks(np.arange(n_parties))
    ax.set_xticklabels(party_labels)
    ax.set_yticklabels(party_labels)

    ax.set_xlabel("outgoing payoff")
    ax.set_ylabel("incoming payoff")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n_parties):
        for j in range(n_parties):
            text = ax.text(j, i, "{:.3f}".format(payoff_mat[i, j]),
                        ha="center", va="center", color="w")

    fig.tight_layout()

    if saved_fn_prefix is not None:
        fig.savefig("{}_payoff.pdf".format(saved_fn_prefix))


def visualize_income(payoff_mat, party_labels, saved_fn_prefix = None, ylim=None):
    fig, ax = plt.subplots(figsize=(4,3))

    outgoing_payoff = np.sum(payoff_mat, axis=1)  # outgoing payoff
    incoming_payoff = np.sum(payoff_mat, axis=0)  # incoming payoff

    ax.bar(party_labels, incoming_payoff, color=colors[0], label="incoming payoff")
    ax.bar(party_labels, -outgoing_payoff, color=colors[1], label="outgoing payoff")
    ax.bar(party_labels, incoming_payoff - outgoing_payoff, color=colors[2], alpha=0.6, label="total payoff")#color=(0., 0., 0., 0.4), edgecolor=(0., 0., 0., 0.8), linewidth=0.0)
    
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.legend()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    fig.tight_layout()
    if saved_fn_prefix is not None:
        fig.savefig("{}_income.pdf".format(saved_fn_prefix))



def visualize_model_reward(payoff_mat, self_values, party_labels, saved_fn_prefix = None):
    n_parties = payoff_mat.shape[0]
    
    outgoing_payoff = np.sum(payoff_mat, axis=1)

    # model reward = self value + outgoing payoff

    fig, ax = plt.subplots(figsize=(4,3))

    ax.bar(party_labels, self_values + outgoing_payoff, color=colors[0], label="outgoing payoff")
    ax.bar(party_labels, self_values, color=colors[1], label=r"$v(i)$")
    
    ax.legend()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    fig.tight_layout()
    if saved_fn_prefix is not None:
        fig.savefig("{}_model.pdf".format(saved_fn_prefix))
