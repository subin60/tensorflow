import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
import urllib.request
import time
from tensorflow import keras


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.97:
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True
callbacks = myCallback()


save_path = "model_experiments_C2W1"
def create_model_checkpoint(model_name="model", save_path=save_path):
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    model_name_with_time = f"{model_name}_{current_time}"

    # Save the model file name to a text file with the same name as save_path
    with open(f'{save_path}.txt', 'a') as file:  # 'a' stands for 'append'
        file.write(os.path.join(save_path, model_name_with_time + ".h5") + '\n')  # add a newline character at the end

    return keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name_with_time + ".h5"),
                                           verbose=0,
                                           save_best_only=True)


create_model_checkpoint = create_model_checkpoint()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir_train = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages_split/training'
base_dir_test = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages_split/validation'


train_datagen = ImageDataGenerator(rescale=1/255.,
                                    rotation_range=20,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,)

train_generator = train_datagen.flow_from_directory(directory=base_dir_train,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='binary',
                                                    seed=42)

test_datagen = ImageDataGenerator(rescale=1/255.)
test_generator = test_datagen.flow_from_directory(directory=base_dir_test,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='binary',
                                                    seed=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='RMSprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)




history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=test_generator,
                    callbacks=[callbacks, create_model_checkpoint]
)


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
    loss, accuracy = model_loaded.evaluate(test_generator)
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