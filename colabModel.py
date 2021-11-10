import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import time 


class ColabModel():

    def __init__(self, dataset_name, party_data_split, model_config):
        # model_config = {"batchsize", n_neurons, activations, n_epochs}

        if dataset_name == "mnist":
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            self.x_train = np.reshape(self.x_train, (-1, 784)) / 255.
            self.x_test = np.reshape(self.x_test, (-1, 784)) / 255.

        self.input_dim = self.x_train.shape[-1]
        self.n_class = len(np.unique(self.y_train))
        
        self.batchsize = model_config["batchsize"]
        self.n_epoch = model_config["n_epoch"]

        self._split_dataset_to_party(party_data_split)
        self._build_model(model_config)

    def _split_dataset_to_party(self, party_data_split):
        self.x_trains, self.y_trains = party_data_split(self.x_train, self.y_train)
        
        self.n_parties = len(self.x_trains)

        # Prepare the test dataset.
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.test_dataset = test_dataset.batch(self.batchsize)

    
    def _get_train_dataset_for_coalition(self, coalition_bitmask, random_seed_to_shuffle=0):
        # Prepare the training dataset.
        coalition_x_train = []
        coalition_y_train = []
        for party in range(self.n_parties):
            if ((1 << party) & coalition_bitmask):    
                coalition_x_train.append( self.x_trains[party])
                coalition_y_train.append(self.y_trains[party])
        coalition_x_train = np.concatenate(coalition_x_train, axis=0)
        coalition_y_train = np.concatenate(coalition_y_train, axis=0)

        np.random.seed(0)
        idxs = list(range(coalition_x_train.shape[0]))
        np.random.shuffle(idxs)

        coalition_x_train = coalition_x_train[idxs]
        coalition_y_train = coalition_y_train[idxs]

        print("coalition_x_train.shape =", coalition_x_train.shape)
        print("coalition_y_train.shape =", coalition_y_train.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((coalition_x_train, coalition_y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batchsize)

        return train_dataset

    def _build_model(self, model_config):
        n_neurons = model_config["n_neurons"]
        activations = model_config["activations"]

        n_hidden_layers = min(len(n_neurons), len(activations))

        inputs = keras.Input(shape=(self.input_dim,), name="input")

        tmp = inputs
        for i in range(n_hidden_layers):
            tmp = tf.keras.layers.Dense(n_neurons[i], activation=activations[i])(tmp)

        outputs = tf.keras.layers.Dense(self.n_class, name="logits")(tmp)

        self.model = keras.Model(inputs=inputs, outputs=outputs) # logits

        # Instantiate an optimizer.
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the metrics.
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    def train(self, coalition_bitmask, init_weights=None):
        print("\n##Train for coalition", bin(coalition_bitmask))
        train_dataset = self._get_train_dataset_for_coalition(coalition_bitmask)

        # Reset training metrics at the end of each epoch
        self.train_acc_metric.reset_states()

        # initialize the weight of the model
        if init_weights is not None:
            for i,weight in enumerate(init_weights):
                self.model.trainable_weights[i].assign(weight)

        for epoch in range(self.n_epoch):
            print("Start of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


                # Update training metric.
                self.train_acc_metric.update_state(y_batch_train, logits)


            #     # Log every 200 batches.
            #     if step % 200 == 0:
            #         print(
            #             "Training loss (for one batch) at step %d: %.4f"
            #             % (step, float(loss_value))
            #         )
            #         print("Seen so far: %s samples" % ((step + 1) * self.batchsize))

            # # Display metrics at the end of each epoch.
            # train_acc = self.train_acc_metric.result()
            # print("Training acc over epoch: %.4f" % (float(train_acc),))

            # # Reset training metrics at the end of each epoch
            # self.train_acc_metric.reset_states()


            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.test_dataset:
                val_logits = self.model(x_batch_val, training=False)
                # Update val metrics
                self.test_acc_metric.update_state(y_batch_val, val_logits)
            test_acc = self.test_acc_metric.result()
            self.test_acc_metric.reset_states()
            print("  Test acc: %.4f" % (float(test_acc),))
            print("  Time taken: %.2fs" % (time.time() - start_time))

        return test_acc
