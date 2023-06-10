import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import random

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)


import requests

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got an HTTP 200 response
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded file: {filename}")

# Usage
url = "https://storage.googleapis.com/tensorflow-1-public/course4/Sunspots.csv"
filename = "./Sunspots.csv"
#download_file(url, filename)


SUNSPOTS_CSV = './Sunspots.csv'

with open(SUNSPOTS_CSV, 'r') as csvfile:
    print(f"Header looks like this:\n\n{csvfile.readline()}")
    print(f"First data point looks like this:\n\n{csvfile.readline()}")
    print(f"Second data point looks like this:\n\n{csvfile.readline()}")

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def parse_data_from_file(filename):
    times = []
    sunspots = []

    with open(filename) as csvfile:


        reader = csv.reader(csvfile, delimiter=',')

        next(reader)
        for i, row in enumerate(reader):
            times.append(i)
            sunspots.append(float(row[2]))


    return times, sunspots

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()


import time
import os
save_path = "model_experiments_C4W4_Sunspots"

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

# Test your function and save all "global" variables within the G class (G stands for global)
@dataclass
class G:
    SUNSPOTS_CSV = './Sunspots.csv'
    times, sunspots = parse_data_from_file(SUNSPOTS_CSV)
    TIME = np.array(times)
    SERIES = np.array(sunspots)
    SPLIT_TIME = int(len(TIME)*0.9)
    WINDOW_SIZE = 30
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

#plt.figure(figsize=(10, 6))
#plot_series(G.TIME, G.SERIES)
#plt.show()

#print(G.SPLIT_TIME)

def train_val_split(time, series, time_step=G.SPLIT_TIME):

    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid


# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(G.TIME, G.SERIES)

def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

# Apply the transformation to the training set
train_set = windowed_dataset(series_train, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)
valid_set = windowed_dataset(series_valid, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)

def create_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[G.WINDOW_SIZE, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])


    return model


# Test your uncompiled model

# Create an instance of the model
uncompiled_model = create_uncompiled_model()


# Get one batch of the training set(X = input, y = label)
for X, y in train_set.take(1):
    # Generate a prediction
    print(f'Testing model prediction with input of shape {X.shape}...')
    y_pred = uncompiled_model.predict(X)

# Compare the shape of the prediction and the label y (remove dimensions of size 1)
y_pred_shape = y_pred.squeeze().shape

assert y_pred_shape == y.shape, (f'Squeezed predicted y shape = {y_pred_shape} '
                                 f'whereas actual y shape = {y.shape}.')

print("Your current architecture is compatible with the windowed dataset! :)")


def adjust_learning_rate(dataset):
    model = create_uncompiled_model()

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))



    # Compile the model passing in the appropriate loss
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                  metrics=["mae"])



    history = model.fit(dataset,
                        epochs=100,
                        callbacks=[lr_schedule])

    return history

#history = adjust_learning_rate(train_set)


import numpy as np
import matplotlib.pyplot as plt

def plot_optimal_learning_rate(history, axis_range=[1e-8, 1e-2, 0, 60]):
    """
    Plots learning rate against loss from a model's history and marks the learning rate with minimum loss.

    Args:
    history: History object from training a keras model with varying learning rates.
    axis_range: list of 4 elements defining the range of x and y axis on the plot.
                Format: [xmin, xmax, ymin, ymax].

    Returns:
    None. Plots a graph of learning rate vs. loss.
    """
    # Extract learning rates and losses from history
    learning_rates = np.array(history.history["lr"])
    losses = np.array(history.history["loss"])

    # Filter out NaN losses
    valid_indices = ~np.isnan(losses)
    learning_rates = learning_rates[valid_indices]
    losses = losses[valid_indices]

    # Identify best learning rate
    best_lr = learning_rates[np.argmin(losses)]
    print(f"Best Learning Rate: {best_lr:.1e}")

    # Create a plot of the learning rate against the loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(history.history["lr"], history.history["loss"])

    # Add a red dot for the best learning rate
    ax.plot(best_lr, np.min(losses), 'ro')

    # Define the range of learning rates that we're interested in
    ax.axis(axis_range)

    # Add secondary x-axis for more detailed ticks
    secax = ax.secondary_xaxis('top')
    secax.set_xscale('log')
    x_ticks = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), num=10)
    secax.set_xticks(x_ticks)
    secax.set_xticklabels([f"{x:.1e}" for x in x_ticks], fontsize=8)  # smaller font size

    # Show the plot
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.grid(True)
    plt.show()

#plot_optimal_learning_rate(history, axis_range=[1e-8, 1e-2, 0, 60])



def create_model():
    model = create_uncompiled_model()

    # Set the learning rate
    learning_rate = 1e-7

    # Set the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])



    return model


model = create_model()


history = model.fit(train_set,
                    epochs=100,
                    validation_data=valid_set,
                    callbacks = [callbacks, create_model_checkpoint])


def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

# Compute the forecast for all the series
rnn_forecast = model_forecast(model, G.SERIES, G.WINDOW_SIZE).squeeze()

# Slice the forecast to get only the predictions for the validation set
rnn_forecast = rnn_forecast[G.SPLIT_TIME - G.WINDOW_SIZE:-1]

# Plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)

mse, mae = compute_metrics(series_valid, rnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")
