import csv
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from scipy.stats import linregress

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()


import time
import os
save_path = "model_experiments_C3W3"

def create_model_checkpoint(model_name="model", save_path=save_path):
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    model_name_with_time = f"{model_name}_{current_time}"

    # Save the model file name to a text file with the same name as save_path
    with open(f'{save_path}.txt', 'a') as file:  # 'a' stands for 'append'
        file.write(os.path.join(save_path, model_name_with_time + ".h5") + '\n')  # add a newline character at the end

    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name_with_time + ".h5"),
                                           verbose=0,
                                           save_best_only=True)

create_model_checkpoint = create_model_checkpoint()


EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = 160000
TRAINING_SPLIT = 0.9



import zipfile
import urllib.request
def download_and_unzip(url, extract_to='.'):
    """
    Downloads and unzips a zip file from a specified URL.

    Args:
    url: A string specifying the URL of the zip file to download.
    extract_to: A string specifying the directory to extract the zip file contents to.

    Returns:
    None.
    """
    # Download the file from the URL
    zip_path, _ = urllib.request.urlretrieve(url)

    # Create a ZipFile object
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all the contents of the zip file in current directory
        zip_ref.extractall(extract_to)

    print(f"Downloaded and extracted file from {url} to {extract_to}")

#download_and_unzip('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip',
                   #'/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/Sentiments')

SENTIMENT_CSV = "./Sentiments/training.1600000.processed.noemoticon.csv"

import pandas as pd

df = pd.read_csv(SENTIMENT_CSV, encoding='ISO-8859-1', header=None)
df.columns = ["polarity", "id", "date", "query", "user", "text"]
# Select only the 'polarity' and 'text' columns
#df_selected = df[['polarity', 'text']]

# Write to a new CSV file, without the header and index
df.to_csv('./Sentiments/training_cleaned.csv', header=False, index=False)

SENTIMENT_CSV = './Sentiments/training_cleaned.csv'

with open(SENTIMENT_CSV, 'r') as csvfile:
    print(f"First data point looks like this:\n\n{csvfile.readline()}")
    print(f"Second data point looks like this:\n\n{csvfile.readline()}")


import csv

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file
    Args:
    filename (string): path to the CSV file
    Returns:
    sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            sentences.append(row[5])
            if row[0] == '4':
                labels.append(1)  # the first column contains labels
            else:
                labels.append(int(row[0]))  # the first column contains labels
    return sentences, labels

# Test your function
sentences, labels = parse_data_from_file(SENTIMENT_CSV)

print(f"dataset contains {len(sentences)} examples\n")
print(f"Text of second example should look like this:\n{sentences[1]}\n")
print(f"Text of fourth example should look like this:\n{sentences[3]}")
print(f"\nLabels of last 5 examples should look like this:\n{labels[-5:]}")

# Bundle the two lists into a single one
sentences_and_labels = list(zip(sentences, labels))
# Perform random sampling
sentences_and_labels = random.sample(sentences_and_labels, MAX_EXAMPLES)
# Unpack back into separate lists
sentences, labels = zip(*sentences_and_labels)
print(f"There are {len(sentences)} sentences and {len(labels)} labels after random sampling\n")



def train_val_split(sentences, labels, training_split):
    """
    Splits the dataset into training and validation sets
    Args:
    sentences (list of string): lower-cased sentences without stopwords
    labels (list of string): list of labels
    training split (float): proportion of the dataset to include in the train set
    Returns:
    train_sentences, validation_sentences, train_labels, validation_labels - lists containing the data splits
    """
    # Compute the number of sentences that will be used for training
    train_size = int(len(sentences) * training_split)

    # Split the sentences and labels into train/validation splits
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    return train_sentences, validation_sentences, train_labels, validation_labels


# Call your function
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

# Print out some results to check if it's working
print(f"There are {len(train_sentences)} sentences for training.")
print(f"There are {len(train_labels)} labels for training.")
print(f"There are {len(val_sentences)} sentences for validation.")
print(f"There are {len(val_labels)} labels for validation.")


from tensorflow.keras.preprocessing.text import Tokenizer

def fit_tokenizer(train_sentences, oov_token):
    """
    Instantiates the Tokenizer class
    Args:
    sentences (list): lower-cased sentences without stopwords
    Returns:
    tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    # Instantiate the Tokenizer class by passing in the oov_token argument
    tokenizer = Tokenizer(oov_token=oov_token)
    # Fit on the sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer


tokenizer = fit_tokenizer(train_sentences, OOV_TOKEN)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)
print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")
print(f"\nindex of word 'i' should be {word_index['i']}")


def seq_pad_and_trunc(sentences, tokenizer, padding, truncating, maxlen):
    """
    Generates an array of token sequences and pads them to the same length
    Args:
    sentences (list of string): list of sentences to tokenize and pad
    tokenizer (object): Tokenizer instance containing the word-index dictionary
    padding (string): type of padding to use
    maxlen (int): maximum length of the token sequence
    Returns:
    padded_sequences (array of int): tokenized sentences padded to the same length
    """

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    # Pad the sequences using the correct padding, truncating and maxlen
    pad_trunc_sequences = pad_sequences(sequences, padding=padding, truncating=truncating, maxlen=maxlen)

    return pad_trunc_sequences

# Test your function
train_pad_trunc_seq = seq_pad_and_trunc(train_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
val_pad_trunc_seq = seq_pad_and_trunc(val_sentences, tokenizer, PADDING, TRUNCATING, MAXLEN)
print(f"Padded training sequences have shape: {train_pad_trunc_seq.shape}\n")
print(f"Padded validation sequences have shape: {val_pad_trunc_seq.shape}")

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

import numpy as np

# Define path to file containing the embeddings
GLOVE_FILE = './Sentiments/glove.6B.100d.txt'

# Initialize an empty embeddings index dictionary
GLOVE_EMBEDDINGS = {}

# Read file and fill GLOVE_EMBEDDINGS with its contents
with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        GLOVE_EMBEDDINGS[word] = coefs

test_word = 'dog'
test_vector = GLOVE_EMBEDDINGS[test_word]
print(f"Vector representation of word {test_word} looks like this:\n\n{test_vector}")

print(f"Each word vector has shape: {test_vector.shape}")

# Initialize an empty numpy array with the appropriate size
EMBEDDINGS_MATRIX = np.zeros((VOCAB_SIZE+1, EMBEDDING_DIM))

# Iterate all of the words in the vocabulary and if the vector representation
# for each word exists within GloVe's representations, save it in the
# EMBEDDINGS_MATRIX array
for word, i in word_index.items():
    embedding_vector = GLOVE_EMBEDDINGS.get(word)
    if embedding_vector is not None:
        EMBEDDINGS_MATRIX[i] = embedding_vector

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from keras.regularizers import l2

def create_model(vocab_size, embedding_dim, maxlen, embeddings_matrix):
    """
    Creates a binary sentiment classifier model
    Args:
    vocab_size (int): size of the vocabulary for the Embedding layer input
    embedding_dim (int): dimensionality of the Embedding layer output
    maxlen (int): length of the input sequences
    embeddings_matrix (array): predefined weights of the embeddings
    Returns:
    model (tf.keras Model): the sentiment classifier model
    """

    model = tf.keras.Sequential([
        # Embedding layer
        tf.keras.layers.Embedding(vocab_size + 1,
                                  embedding_dim,
                                  input_length=maxlen,
                                  weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        # Bidirectional LSTM with 64 units
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # Bidirectional LSTM with 64 units
        tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with 64 units and ReLU activation
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# Create your untrained model
model = create_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN, EMBEDDINGS_MATRIX)

# Train the model and save the training history
history = model.fit(train_pad_trunc_seq,
                    train_labels,
                    epochs=10,
                    validation_data=(val_pad_trunc_seq, val_labels),
                    callbacks=[callbacks, create_model_checkpoint]
                    )



def evaluate_models(save_path, val_padded_seq, val_labels):
    # Read the model file names from the text file
    with open(f'{save_path}.txt', 'r') as file:
        model_paths = file.read().splitlines()

    accuracies = []
    losses = []
    timestamps = []

    # Evaluate each model
    for model_path in model_paths:
        print(f'Evaluating model: {model_path}')
        model_loaded = tf.keras.models.load_model(model_path)

        # Check if input shape matches
        expected_input_shape = model_loaded.layers[0].input_shape
        if expected_input_shape[1] != len(val_padded_seq[0]):
            print(f'Skipping {model_path} due to input shape mismatch: expected {expected_input_shape}, got {val_padded_seq.shape}.')
            continue

        # Evaluate model
        try:
            loss, accuracy = model_loaded.evaluate(val_padded_seq, val_labels)
            print(f'Loss: {loss}, Accuracy: {accuracy}')
            accuracies.append(accuracy)
            losses.append(loss)

            # Extract only the time from the filename
            time = model_path.split('_')[-1].split('.')[0].split('-')[-1]  # 'HH_MM_SS'
            timestamps.append(time)

        except Exception as e:
            print(f'Error evaluating {model_path}: {str(e)}')

    return accuracies, losses, timestamps

accuracies, losses, timestamps = evaluate_models(save_path, val_pad_trunc_seq, val_labels)




import matplotlib.pyplot as plt

def plot_model_accuracies_and_losses(accuracies, losses, timestamps):
    plt.figure(figsize=(10, 5))

    # Plot the accuracy of each model
    plt.subplot(1, 2, 1)
    plt.plot(accuracies)
    plt.title('Model accuracies')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(timestamps)), timestamps, rotation='vertical')  # Set xticks to be the times

    # Plot the loss of each model
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Model losses')
    plt.ylabel('Loss')
    plt.xticks(range(len(timestamps)), timestamps, rotation='vertical')  # Set xticks to be the times

    plt.tight_layout()
    plt.show()

plot_model_accuracies_and_losses(accuracies, losses, timestamps)

def plot_training_validation(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


plot_training_validation(history)