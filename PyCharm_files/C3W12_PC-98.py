import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import requests
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import random

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
save_path = "model_experiments_C3W12"

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




def download_file(url, filename):
    # Send a HTTP request to the specified URL and save the response from server in a response object called r
    if not os.path.exists('bbc'):
        os.makedirs('bbc')

    r = requests.get(url)

    # Open the file in write mode (binary) and write the contents of the response to it
    with open(filename, 'wb') as f:
        f.write(r.content)

# Call the function
#download_file("https://raw.githubusercontent.com/mdsohaib/BBC-News-Classification/master/bbc-text.csv", './bbc/bbc-text.csv')

def print_file_header_and_first_line(filepath):
    with open(filepath, 'r') as csvfile:
        print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
        print(f"Each data point looks like this:\n\n{csvfile.readline()}")

# Call the function
#print_file_header_and_first_line("./bbc/bbc-text.csv")


NUM_WORDS = 10000
EMBEDDING_DIM = 16
MAXLEN = 1032
PADDING = 'post'
OOV_TOKEN = '<OOV>'
TRAINING_SPLIT = .8


def remove_stopwords(sentence):
    """
    Removes a list of stopwords
    Args:
    sentence (string): sentence to remove the stopwords from
    Returns:
    sentence (string): lowercase sentence without the stopwords
    """
    stopwords = ["a", "about", "above", "after", "again", "against", "all",
                 "am", "an", "and", "any", "are", "as", "at", "be", "because",
                 "been", "before", "being", "below", "between", "both", "but",
                 "by", "could", "did", "do", "does", "doing", "down", "during",
                 "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
                 "hers", "herself", "him", "himself", "his", "how", "how's", "i",
                 "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it",
                 "it's", "its", "itself", "let's", "me", "more", "most", "my",
                 "myself", "nor", "of", "on", "once", "only", "or", "other",
                 "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                 "she", "she'd", "she'll", "she's", "should", "so", "some", "such",
                 "than", "that", "that's", "the", "their", "theirs", "them",
                 "themselves", "then", "there", "there's", "these", "they",
                 "they'd", "they'll", "they're", "they've", "this", "those",
                 "through", "to", "too", "under", "until", "up", "very", "was",
                 "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's",
                 "when", "when's", "where", "where's", "which", "while", "who",
                 "who's", "whom", "why", "why's", "with", "would", "you", "you'd",
                 "you'll", "you're", "you've", "your", "yours", "yourself",
                 "yourselves"]

    sentence = sentence.lower()
    words = sentence.split()
    words = [word for word in words if word not in stopwords]
    sentence = ' '.join(words)
    return sentence

#print(remove_stopwords('I am about to go to the store and get any snack'))


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
        reader = csv.reader(csvfile)
        next(reader) # Skip the header row
        for row in reader:
            labels.append(row[0])
            sentence = remove_stopwords(row[1]) # Use the remove_stopwords function
            sentences.append(sentence)

    return sentences, labels

sentences, labels = parse_data_from_file('./bbc/bbc-text.csv')


print("ORIGINAL DATASET:\n")
print(f"There are {len(sentences)} sentences in the dataset.")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).")
print(f"There are {len(labels)} labels in the dataset.")
print(f"The first 5 labels are {labels[:5]}\n\n")


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

def fit_tokenizer(train_sentences, num_words, oov_token):
    """
    Instantiates the Tokenizer class
    Args:
    sentences (list): lower-cased sentences without stopwords
    Returns:
    tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary
    """
    # Instantiate the Tokenizer class by passing in the oov_token argument
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    # Fit on the sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer


tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)
word_index = tokenizer.word_index
print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")


from tensorflow.keras.preprocessing.sequence import pad_sequences


def seq_and_pad(sentences, tokenizer, padding, maxlen):
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
    # Pad the sequences using the correct padding and maxlen
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)

    return padded_sequences

# Test your function
train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)
print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
print(f"Padded validation sequences have shape: {val_padded_seq.shape}")


# Define the function
def tokenize_labels(all_labels, split_labels):
    """
    Tokenizes the labels.
    Args:
    all_labels (list of string): labels to generate the word-index from
    split_labels (list of string): labels to tokenize
    Returns:
    label_seq_np (numpy array of int): tokenized labels
    """

    # Instantiate the Tokenizer (no additional arguments needed)
    label_tokenizer = Tokenizer()
    # Fit the tokenizer on all the labels
    label_tokenizer.fit_on_texts(all_labels)
    # Convert labels to sequences
    label_seq = label_tokenizer.texts_to_sequences(split_labels)

    # Convert sequences to a numpy array. We subtract 1 from every entry in
    # the array to shift from 1-based indexing to 0-based indexing.
    label_seq_np = np.array(label_seq) - 1

    return label_seq_np

# Test your function
train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

print("First 5 labels of the training set should look like this:\n", train_label_seq[:5])
print("\nFirst 5 labels of the validation set should look like this:\n", val_label_seq[:5])
print("\nTokenized labels of the training set have shape:", train_label_seq.shape)
print("\nTokenized labels of the validation set have shape:", val_label_seq.shape)


def create_model(num_words, embedding_dim, maxlen):
    """
    Creates a text classifier model
    Args:
    num_words (int): size of the vocabulary for the Embedding layer input
    embedding_dim (int): dimensionality of the Embedding layer output
    maxlen (int): length of the input sequences
    Returns:
    model (tf.keras Model): the text classifier model
    """
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)

history = model.fit(train_padded_seq,
                    train_label_seq,
                    epochs=50,
                    validation_data=(val_padded_seq, val_label_seq),
                    callbacks=[callbacks, create_model_checkpoint])



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
        loss, accuracy = model_loaded.evaluate(val_padded_seq, val_label_seq)
        print(f'Loss: {loss}, Accuracy: {accuracy}')
        accuracies.append(accuracy)
        losses.append(loss)

        # Extract only the time from the filename
        time = model_path.split('_')[-1].split('.')[0].split('-')[-1]  # 'HH_MM_SS'
        timestamps.append(time)

    except Exception as e:
        print(f'Error evaluating {model_path}: {str(e)}')




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
