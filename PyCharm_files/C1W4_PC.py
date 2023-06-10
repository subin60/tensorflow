import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
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

download_and_unzip('https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip',
                   '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments')
tf.random.set_seed(42)

base_dir_train = "./pizza_steak/train"
base_dir_test = "./pizza_steak/test"
pizza_dir = os.path.join(base_dir_train, "pizza/")
steak_dir = os.path.join(base_dir_train, "steak/")

#print("Sample pizza image:")
#plt.imshow(load_img(f"{os.path.join(pizza_dir, os.listdir(pizza_dir)[0])}"))
#plt.show()
#print("\nSample steak image:")
#plt.imshow(load_img(f"{os.path.join(steak_dir, os.listdir(steak_dir)[0])}"))
#plt.show()

from tensorflow.keras.preprocessing.image import img_to_array
# Load the first example of a happy face
sample_image = load_img(f"{os.path.join(pizza_dir, os.listdir(pizza_dir)[0])}")
# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.97:
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True
callbacks = myCallback()


save_path = "model_experiments_C1W4"
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


train_datagen = ImageDataGenerator(rescale=1/255.)
train_generator = train_datagen.flow_from_directory(directory=base_dir_train,
                                                    target_size=(224,224),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    seed=42)

test_datagen = ImageDataGenerator(rescale=1/255.)
test_generator = test_datagen.flow_from_directory(directory=base_dir_test,
                                                    target_size=(224,224),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    seed=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=test_generator,
                    validation_steps=len(test_generator),
                    epochs=10,
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