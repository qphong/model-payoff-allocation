# custom training loop 
# taken from: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time 

"""
collaboration(name, val_size, parties, model_config)
    parties: list of data index in the training dataset
        (note that the training dataset is not shuffled)
    model_config: n_neurons and activation of each layer
        number of training epochs

    construct best_initializer by training the whole model using all training dataset

    get_coalition_value(coalition)
        coalition: list of parties
        return the validation accuracy after training with the coalition's data
            from the best_initializer

    get_best_model_for_party(party, desired_val_acc)
        party: index of the party obtaining this model
        desired_val_acc: the desired val acc of the returned model
            should be >= get_coalition_value([party])
        this is done by training the model initialized at the best_initializer
            using only the data from party!

model(input_dim, nclass, n_neurons)
    construct the model
    train(training_dataset, validation_dataset)
        training_dataset is constructed from the parties' data in the coalition
"""


inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs) # logits


# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)) / 255.
x_test = np.reshape(x_test, (-1, 784)) / 255.

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)


# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = keras.metrics.SparseCategoricalAccuracy()


epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
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
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)


        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))

    # Run a testing loop at the end of each epoch.
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test, training=False)
        # Update test metrics
        test_acc_metric.update_state(y_batch_test, test_logits)
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print("Test acc: %.4f" % (float(test_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))



# class Model():
#     def __init__(self):
#         mnist = tf.keras.datasets.mnist

#         (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
#         self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

#         self.model = tf.keras.models.Sequential([
#             tf.keras.layers.Flatten(input_shape=(28, 28)),
#             tf.keras.layers.Dense(128, activation='relu'),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(10)
#         ])

#         self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#         self.model.compile(optimizer='adam',
#               loss=self.loss_fn,
#               metrics=['accuracy'])

#     def train(self):
#         self.model.fit(self.x_train, self.y_train, epochs=5)
#         self.model.evaluate(self.x_test, self.y_test, verbose=2)

#     def predict(self, x_predict):
#         predictions = self.model(x_predict).numpy()
#         return tf.nn.softmax(predictions).numpy()
