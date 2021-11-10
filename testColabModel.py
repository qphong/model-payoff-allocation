import numpy as np
import tensorflow as tf
import pickle
import sys

import colabModel


config_name = "6_parties_replication_1"
party_digits_config = {
    "5_non_overlap_parties": [[0,1], [2,3], [4,5], [6,7], [8,9]],
    "5_parties_no_replication": [[0], [1, 2], [0, 1, 2, 3], [3, 4, 5], [6, 7, 8, 9]],
    "5_parties_no_replication_1all": [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2],
        [0, 1, 2, 3],
        [3, 4, 5],
        [6, 7, 8, 9],
    ],
    "6_parties_replication_1": [
        [0,],
        [1, 2],
        [1, 2],
        [0, 1, 2, 3],
        [3, 4, 5],
        [6, 7, 8, 9],
    ],
    "6_parties_replication_4": [
        [0],
        [1, 2],
        [0, 1, 2, 3],
        [3, 4, 5],
        [6, 7, 8, 9],
        [6, 7, 8, 9],
    ],
}

party_digits = party_digits_config[config_name]
print("config_name:", config_name)
print(party_digits)


def mnist_parties_data_split(x_train, y_train):
    # split data based on the digits
    # [0], [1,2], [3,4,5], [6,7,8,9]
    x_trains = []
    y_trains = []
    for party in range(len(party_digits)):

        party_data_idxs = []
        for digit in party_digits[party]:
            idxs = np.where(y_train == digit)[0]
            party_data_idxs.extend(list(idxs))

        party_x_train = x_train[party_data_idxs]
        party_y_train = y_train[party_data_idxs]
        x_trains.append(party_x_train)
        y_trains.append(party_y_train)

    return x_trains, y_trains


def test_mnist_parties_data_split():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    x_trains, y_trains = mnist_parties_data_split(x_train, y_train)

    for i, ys in enumerate(y_trains):
        print("Party", i)
        print(ys.shape)
        print(np.unique(ys))


model = colabModel.ColabModel(
    dataset_name="mnist",
    party_data_split=mnist_parties_data_split,
    model_config={
        "n_epoch": 5,
        "batchsize": 64,
        "n_neurons": [64, 64],
        "activations": ["relu", "relu"],
    },
)

# init_weights = model.model.get_weights()

# with open("mnist_init_weights.pkl", "wb") as outfile:
#     pickle.dump(init_weights, outfile, protocol=pickle.HIGHEST_PROTOCOL)


with open("mnist_init_weights.pkl", "rb") as infile:
    init_weights = pickle.load(infile)

n_party = len(party_digits)
test_accs = np.zeros(1 << n_party)

for i in range(1, 1 << n_party):
    test_accs[i] = model.train(i, init_weights=init_weights)
    sys.stdout.flush()

with open("mnist_v_{}.pkl".format(config_name), "wb") as outfile:
    pickle.dump(
        {"party_digits": party_digits, "name": config_name, "test_acc": test_accs},
        outfile,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
