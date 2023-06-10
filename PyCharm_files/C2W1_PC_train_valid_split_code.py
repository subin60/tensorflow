import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
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

download_and_unzip('https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip',
                   '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments')
tf.random.set_seed(42)

source_path = 'PetImages'
source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')
# Deletes all non-image files (there are two .db files bundled into the dataset)
os.system(f'find {source_path} -type f ! -name "*.jpg" -exec rm {{}} +')
# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
# Load the first example of a happy face
sample_image = load_img(f"{os.path.join(source_path_dogs, os.listdir(source_path_dogs)[0])}")
# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

import os
import random
import tensorflow as tf
from shutil import copyfile
import shutil


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE_DIR):
        file = SOURCE_DIR + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE_DIR + filename
        destination = TRAINING_DIR + filename
        copyfile(this_file, destination)

    for filename in validation_set:
        this_file = SOURCE_DIR + filename
        destination = VALIDATION_DIR + filename
        copyfile(this_file, destination)


def clear_directory(directory):
    """
    Clears all files in the given directory
    Args:
    directory (str): path to the directory to clear
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# Setting seed to get consistent results
tf.random.set_seed(42)

CAT_SOURCE_DIR = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages/Cat/'
DOG_SOURCE_DIR = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages/Dog/'
TRAINING_DIR = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages_split/training/'
VALIDATION_DIR = '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/PetImages_split/validation/'

TRAINING_CATS_DIR = TRAINING_DIR + 'Cat/'
TRAINING_DOGS_DIR = TRAINING_DIR + 'Dog/'
VALIDATION_CATS_DIR = VALIDATION_DIR + 'Cat/'
VALIDATION_DOGS_DIR = VALIDATION_DIR + 'Dog/'

os.makedirs(TRAINING_CATS_DIR, exist_ok=True)
os.makedirs(TRAINING_DOGS_DIR, exist_ok=True)
os.makedirs(VALIDATION_CATS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DOGS_DIR, exist_ok=True)

# Clear the training and validation directories before splitting
clear_directory(TRAINING_DIR)
clear_directory(VALIDATION_DIR)

# Recreate the necessary directories after clearing
os.makedirs(TRAINING_CATS_DIR, exist_ok=True)
os.makedirs(TRAINING_DOGS_DIR, exist_ok=True)
os.makedirs(VALIDATION_CATS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DOGS_DIR, exist_ok=True)

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

print("Training cats:", len(os.listdir(TRAINING_CATS_DIR)))
print("Validation cats:", len(os.listdir(VALIDATION_CATS_DIR)))
print("Training dogs:", len(os.listdir(TRAINING_DOGS_DIR)))
print("Validation dogs:", len(os.listdir(VALIDATION_DOGS_DIR)))