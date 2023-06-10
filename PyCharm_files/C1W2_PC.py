import os
import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


print(f'X_train Shape: {X_train.shape}')
print(f'y_train Shape: {y_train.shape}')
print(f'X_test Shape: {X_test.shape}')
print(f'y_test Shape: {y_test.shape}')

print("Number of classes in training set: ", len(np.unique(y_train)))
print("Number of classes in test set: ", len(np.unique(y_test)))


X_train = X_train/255.
X_test = X_test/255.

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
        print("\nReached 99% accuracy so cancelling training!")
        self.model.stop_training = True
callbacks = myCallback()
def create_model_checkpoint(model_name="model", save_path="model_experiments"):
    # Create a timestamp string
    current_time = time.strftime("%Y_%m_%d-%H_%M_%S")  # e.g., 2022_02_01-14_05_32

    # Combine model name and timestamp
    model_name_with_time = f"{model_name}_{current_time}"

    return keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name_with_time + ".h5"),
                                           verbose=0,
                                           save_best_only=True)



model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test,y_test),
                    epochs=100,
                    callbacks=[callbacks, create_model_checkpoint()])





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




