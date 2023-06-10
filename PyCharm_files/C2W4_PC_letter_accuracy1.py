import csv
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

import os
import gdown

import os
import gdown

def download_data():
    # Ensure the directory exists
    if not os.path.exists('sign_mnist'):
        os.makedirs('sign_mnist')

    # Now proceed with the downloads
    gdown.download('https://drive.google.com/uc?id=1z0DkA9BytlLxO1C0BAWzknLyQmZAp0HR', 'sign_mnist/sign_mnist_train.csv', quiet=False)
    gdown.download('https://drive.google.com/uc?id=1z1BIj4qmri59GWBG4ivMNFtpZ4AXIbzg', 'sign_mnist/sign_mnist_test.csv', quiet=False)

# Call the function
#download_data()

tf.random.set_seed(42)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()


import time

save_path = "model_experiments_C2W4"

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



TRAINING_FILE = './sign_mnist/sign_mnist_train.csv'
VALIDATION_FILE = './sign_mnist/sign_mnist_test.csv'

with open(TRAINING_FILE) as training_file:
    line = training_file.readline()
    print(f"First line (header) looks like this:\n{line}")
    line = training_file.readline()
    print(f"Each subsequent line (data points) look like this:\n{line}")
    pixels_and_label = len(line.split(','))  # Total count including the label
    number_of_pixels = pixels_and_label - 1  # Subtract one for the label
    print(f'Number of pixels {number_of_pixels}')

def parse_data_from_input(filename):

    with open(filename) as file:
        # load the csv file as a numpy array, and skip the first row (header)
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        # separate the labels (first column) and image data (rest of the columns)
        labels = data[:, 0].astype(np.float64)
        images = data[:, 1:].astype(np.float64)
        # reshape the image data to 28x28
        images = images.reshape(-1, 28, 28)

    return images, labels

training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")

# Plot a sample of 10 images from the training set
def plot_categories(training_images, training_labels):
    fig, axes = plt.subplots(1, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)
    for k in range(10):
        img = training_images[k]
        img = np.expand_dims(img, axis=-1)
        img = array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap="Greys_r")
        ax.set_title(f"{letters[int(training_labels[k])]}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

#plot_categories(training_images, training_labels)

def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    # Convert images to float32
    training_images = training_images.astype(np.float32)
    validation_images = validation_images.astype(np.float32)

    # Add a new axis for the color channel
    training_images = np.expand_dims(training_images, axis=-1)
    validation_images = np.expand_dims(validation_images, axis=-1)

    train_datagen = ImageDataGenerator(rescale = 1/255.,
                                       rotation_range=18,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.0,
                                       zoom_range=0.20,
                                       horizontal_flip=True,)
                                       #fill_mode='nearest')

    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=32,
                                         seed=42)

    test_datagen = ImageDataGenerator(rescale = 1/255.)

    test_generator = test_datagen.flow(x=validation_images,
                                       y=validation_labels,
                                       batch_size=32,
                                       seed=42)

    return train_generator, test_generator

train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

print(f"Images of training generator have shape: {train_generator.x.shape} and dtype: {train_generator.x.dtype}")
print(f"Labels of training generator have shape: {train_generator.y.shape}) and dtype: {train_generator.y.dtype}")
print(f"Images of validation generator have shape: {validation_generator.x.shape} and dtype: {validation_generator.x.dtype}")
print(f"Labels of validation generator have shape: {validation_generator.y.shape} and dtype: {validation_generator.y.dtype}")


def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=['accuracy']
    )

    return model

model = create_model()

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(validation_generator),
                    epochs=15,
                    verbose=2,
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
    loss, accuracy = model_loaded.evaluate(validation_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    accuracies.append(accuracy)
    losses.append(loss)

    # Extract only the time from the filename
    time = model_path.split('_')[-1].split('.')[0].split('-')[-1]  # 'HH_MM_SS'
    timestamps.append(time)

# Plot the accuracy of each model
plt.figure(figsize=(10,5))
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


# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
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