import numpy as np 
import tensorflow as tf 
import pickle 

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns

colors = sns.color_palette("tab10")

import allocation 



def visualize_payoff(payoff_mat, party_labels, saved_fn_prefix):
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
    im = ax.imshow(payoff_mat, cmap=sns.color_palette("dark:salmon_r", as_cmap=True))
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

    fig.savefig("{}_payoff.pdf".format(saved_fn_prefix))
    # plt.show()

def visualize_income(payoff_mat, party_labels, saved_fn_prefix, ylim=None):
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
    fig.savefig("{}_income.pdf".format(saved_fn_prefix))
    # plt.show()


def visualize_model_reward(payoff_mat, self_values, party_labels, saved_fn_prefix):
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
    fig.savefig("{}_model.pdf".format(saved_fn_prefix))
    # plt.show()


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

name = "5_parties_no_replication"
replicated_party_idxs = []
ylim = (-1.6, 2)

# name = "5_non_overlap_parties"
# replicated_party_idxs = []
# ylim = None

# name = "6_parties_replication_1"
# replicated_party_idxs = [[1,2]]
# ylim = (-1.6, 2)

# name = "6_parties_replication_4"
# replicated_party_idxs = [[4,5]]
# ylim = (-1.6, 2)

with open("mnist_v_{}.pkl".format(name), "rb") as infile:
    mnist_v = pickle.load(infile)

n = len(mnist_v["party_digits"])

value_func = lambda c: mnist_v["test_acc"][c]
party_labels = [str(p) for p in mnist_v["party_digits"]]

unconstrained_payoff, unconstrained_income, unconstrained_model_reward = allocation.get_payoff_flow(n, value_func, budget=None)


# unconstrained_payoff, party_labels = combine_replicates(unconstrained_payoff, replicated_party_idxs, party_labels)

constrained_payoff, constrained_income, constrained_model_reward = allocation.get_payoff_flow(n, value_func, budget=np.zeros(n))

unconstrained_payoff = combine_replicate_payoff(unconstrained_payoff, replicated_party_idxs)
constrained_payoff = combine_replicate_payoff(constrained_payoff, replicated_party_idxs)

party_labels = remove_replicate_party_labels(party_labels, replicated_party_idxs)

self_values = np.zeros(n)
for i in range(n):
    self_values[i] = value_func(1 << i)
    
self_values = remove_replicate_party_self_values(self_values, replicated_party_idxs)

saved_fn_prefix = "img/{}_unconstrained".format(mnist_v["name"])

visualize_payoff(unconstrained_payoff, party_labels, saved_fn_prefix)

visualize_income(unconstrained_payoff, party_labels, saved_fn_prefix, ylim)

print("TODO: Model reward is not correct for duplicated party!")
visualize_model_reward(unconstrained_payoff, self_values, party_labels, saved_fn_prefix)

saved_fn_prefix = "img/{}_constrained".format(mnist_v["name"])

visualize_payoff(constrained_payoff, party_labels, saved_fn_prefix)

visualize_income(constrained_payoff, party_labels, saved_fn_prefix, ylim)

visualize_model_reward(constrained_payoff, self_values, party_labels, saved_fn_prefix)

# plt.show()
