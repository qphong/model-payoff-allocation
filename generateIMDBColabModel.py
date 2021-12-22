"""
Sentiment analysis on the IMDB dataset
"""

from typing import Protocol
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pickle

from tensorflow.keras import layers
from tensorflow.keras import losses


# Download the IMDB dataset
# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file("aclImdb_v1", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')

# dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# # only use aclImdb/train/pos and aclImdb/train/neg
# train_dir = os.path.join(dataset_dir, 'train')
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)

batch_size = 32
seed = 0

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)


# standardize, tokenize, and vectorize the data
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("156 ---> ",vectorize_layer.get_vocabulary()[156])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)



# Create the model
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# train_ds size: 20000
# val_ds size: 5000
"""
There are 5 different parties with dataset:
0 ----> 7k
   1k ----> 8k
            8k ----> 12k
                     12k --------> 20k
"""
name = "imdb_5_parties"
n_parties = 5
datasets = [None] * n_parties # list of datasets, each for 1 party

# given batch_size = 32
# there are 625 batches
if name == "imdb_4_parties":
    datasets[0] = train_ds.take(218)
    datasets[1] = train_ds.skip(31).take(218)
    datasets[2] = train_ds.skip(218+31).take(125)
    datasets[3] = train_ds.skip(343)
elif name == "imdb_4_parties_1all":
    datasets[0] = train_ds.skip(0)
    datasets[1] = train_ds.take(218)
    datasets[2] = train_ds.skip(218).take(125)
    datasets[3] = train_ds.skip(218+125)
elif name == "imdb_5_parties":
    ndata = int(625/5)
    datasets[0] = train_ds.take(ndata)
    datasets[1] = train_ds.skip(ndata).take(ndata)
    datasets[2] = train_ds.skip(ndata*2).take(ndata)
    datasets[3] = train_ds.skip(ndata*3).take(ndata)
    datasets[4] = train_ds.skip(ndata*4).take(ndata)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

for bitmask in range(1, 1 << n_parties):
    print("\n##Train for coalition", bin(bitmask))

    # merge datasets for coalition with bitmask
    first = True
    cur_ds = None
    for i in range(n_parties):
        if (1 << i) & bitmask:
            cur_ds = datasets[i] if first else cur_ds.concatenate(datasets[i])
            first = False

    # cur_ds = cur_ds.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False).batch(32)

    cur_ds = cur_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # init_weights = model.get_weights()
    
    # with open("imdb_sentiment_analysis/imdb_init_weights.pkl", "wb") as outfile:
    #     pickle.dump(init_weights, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    with open("imdb_sentiment_analysis/imdb_init_weights.pkl", "rb") as infile:
        init_weights = pickle.load(infile)

    for i,weight in enumerate(init_weights):
        model.trainable_weights[i].assign(weight)

    epochs = 10
    history = model.fit(
        cur_ds,
        validation_data=val_ds,
        epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)

    print("\tLoss: ", loss)
    print("\tAccuracy: ", accuracy)

    with open("imdb_sentiment_analysis/{}.pkl".format(name), "wb") as outfile:
        pickle.dump({
            "party_labels": [str(i) for i in range(n_parties)],
            "name": name,
            "test_acc": accuracy,
            "replicated_party_idxs": []
        },
        outfile,
        protocol=pickle.HIGHEST_PROTOCOL)
