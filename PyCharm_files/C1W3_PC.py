import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f'X_train Shape: {X_train.shape}')
print(f'y_train Shape: {y_train.shape}')
print(f'X_test Shape: {X_test.shape}')
print(f'y_test Shape: {y_test.shape}')

print("Number of classes in training set: ", len(np.unique(y_train)))
print("Number of classes in test set: ", len(np.unique(y_test)))


def reshape_and_normalize(images):
    images = images.reshape(images.shape + (1,))
    images = images/255.
    return images

X_train = reshape_and_normalize(X_train)
X_test = reshape_and_normalize(X_test)

print(f'X_train Shape, after reshape: {X_train.shape}')
print(f'y_train Shape: {y_train.shape}')
print(f'X_test Shape, after reshape: {X_test.shape}')
print(f'y_test Shape: {y_test.shape}')

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True
callbacks = myCallback()

# Rest of your code...

def create_model_checkpoint(model_name="model", save_path="model_experiments"):
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    model_name_with_time = f"{model_name}_{current_time}"

    # Save the model file name to a text file
    with open('model_filenames.txt', 'a') as file:  # 'a' stands for 'append'
        file.write(os.path.join(save_path, model_name_with_time + ".h5") + '\n')  # add a newline character at the end

    return keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name_with_time + ".h5"),
                                           verbose=0,
                                           save_best_only=True)

# Rest of your code...


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2),
    #tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #tf.keras.layers.MaxPool2D(2),
    #tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(x=X_train,
                    y=y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    callbacks=[callbacks, create_model_checkpoint()])



# Read the model file names from the text file
with open('model_filenames.txt', 'r') as file:
    model_paths = file.read().splitlines()

accuracies = []
losses = []
timestamps = []

# Evaluate each model
for model_path in model_paths:
    print(f'Evaluating model: {model_path}')
    model_loaded = tf.keras.models.load_model(model_path)
    loss, accuracy = model_loaded.evaluate(X_test, y_test)
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