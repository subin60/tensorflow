
import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

tf.random.set_seed(42)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()


import time

save_path = "model_experiments_C2W3"

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

train_datagen = ImageDataGenerator(1/255.)

train_data = train_datagen.flow_from_directory(directory='horse-or-human/training',
                                               batch_size=32,
                                               class_mode='binary',
                                               target_size=(150, 150))

test_datagen = ImageDataGenerator(1/255.)

test_data = train_datagen.flow_from_directory(directory='horse-or-human/validation',
                                               batch_size=32,
                                               class_mode='binary',
                                               target_size=(150, 150))



import urllib.request

# Define URL and filename
url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
filename = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Download the file from the URL
urllib.request.urlretrieve(url, filename)

print("Weights downloaded successfully.")

from tensorflow.keras.applications.inception_v3 import InceptionV3

# Path to the downloaded weights
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

def create_pre_trained_model(local_weights_file):


    pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                    include_top = False,
                                    weights = None)
    pre_trained_model.load_weights(local_weights_file)
    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False

    return pre_trained_model

# Load the weights you downloaded
pre_trained_model = create_pre_trained_model(local_weights_file)

pre_trained_model.summary()

total_params = pre_trained_model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])
print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")



def output_of_last_layer(pre_trained_model):
    """
    Gets the last layer output of a model
    Args:
    pre_trained_model (tf.keras Model): model to get the last layer output from
    Returns:
    last_output: output of the model's last layer
    """
    ### START CODE HERE
    last_desired_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_desired_layer.output_shape)
    last_output = last_desired_layer.output
    print('last layer output: ', last_output)
    ### END CODE HERE
    return last_output

last_output = output_of_last_layer(pre_trained_model)

def create_final_model(pre_trained_model, last_output):
    x = tf.keras.layers.Flatten()(last_output)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  metrics=['accuracy'])

    return(model)

model = create_final_model(pre_trained_model, last_output)
# Inspect parameters
total_params = model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")


history = model.fit(train_data,
                    validation_data = test_data,
                    epochs = 10,
                    verbose = 2,
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
    loss, accuracy = model_loaded.evaluate(test_data)
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