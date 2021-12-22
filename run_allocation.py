import numpy as np 
import tensorflow as tf 
import pickle 

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns

colors = sns.color_palette("tab10")

import allocation 
import utils

def run_allocation(info_path, budget, budget_id,ylim=None):
    # experiment_name = "mnist_v_5_parties_no_replication"

    # info_path = "mnist_training/{}".format(experiment_name)

    # # budget = None: unlimited budget
    # # otherwise, budget is a vector of size n_party: party[i] is the maximum expense for party i
    # budget = None 
    # budget_id = None
    path_prefix = "img/{}_{}_test".format(experiment_name, budget_id)
    # ylim = (-1.6, 2)

    n_party, value_func, party_labels, replicated_party_idxs = utils.get_collaboration_information(info_path)

    payoff, income, model_reward = allocation.get_payoff_flow(n_party, value_func, budget)

    print("Payoff")
    print(payoff)

    # TODO: implement the case of replicated parties

    self_values = np.zeros(n_party)
    for i in range(n_party):
        self_values[i] = value_func(1 << i)

    utils.visualize_payoff(payoff, party_labels, path_prefix)

    utils.visualize_income(payoff, party_labels, path_prefix, ylim)

    utils.visualize_model_reward(payoff, self_values, party_labels, path_prefix)

if __name__ == "__main__":
    # experiment_name = "mnist_v_5_parties_no_replication"

    # info_path = "mnist_training/{}".format(experiment_name)

    # otherwise, budget is a vector of size n_party: party[i] is the maximum expense for party i
    # budget_id = "unlimited"
    # budget_id = "uniform_zero"
    # budget_id = "first_0_5"
    # budget_id = "first_unlimited"
    # budget_id = "first_two_0_5"
    # budget_id = "first_two_0_25"
    # budget_id = "first_two_unlimited"
    # budget_id = "last_unlimited"

    # if budget_id == "unlimited":
    #     budget = None 
    # elif budget_id == "uniform_zero":
    #     budget = np.zeros(5)
    # elif budget_id == "first_0_5":
    #     budget = np.zeros(5)
    #     budget[0] = 0.5
    # elif budget_id == "first_unlimited":
    #     budget = np.zeros(5)
    #     budget[0] = 1e9
    # elif budget_id == "last_unlimited":
    #     budget = np.zeros(5)
    #     budget[-1] = 1e9
    # elif budget_id == "first_two_0_5":
    #     budget = np.zeros(5)
    #     budget[0] = budget[1] = 0.5
    # elif budget_id == "first_two_0_25":
    #     budget = np.zeros(5)
    #     budget[0] = budget[1] = 0.25
    # elif budget_id == "first_two_unlimited":
    #     budget = np.zeros(5)
    #     budget[0] = budget[1] = 1e9

    # ylim = (-1.6, 2)

    experiment_name = "imdb_5_parties"

    info_path = "imdb_sentiment_analysis/{}".format(experiment_name)

    # budget_id = "unlimited"
    # budget_id = "uniform_zero"
    budget_id = "unlimited_2"

    if budget_id == "unlimited":
        budget = None
    elif budget_id == "uniform_zero":
        budget = np.zeros(5)
    elif budget_id == "unlimited_2":
        budget = np.zeros(5)
        budget[2] = 100. 

    ylim = None

    run_allocation(info_path, budget, budget_id, ylim)
    
    plt.show()