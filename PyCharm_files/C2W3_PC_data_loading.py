import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

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

download_and_unzip('https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip',
                   '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/horse-or-human/training')

download_and_unzip('https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip',
                   '/Users/subinjosethomas/Desktop/Tensorflow_Coursera_Assignments/horse-or-human/validation')

tf.random.set_seed(42)

source_path = 'horse-or-human/training'
source_path_horses = os.path.join(source_path, 'horses')
source_path_humans = os.path.join(source_path, 'humans')
# Deletes all non-image files (there are two .db files bundled into the dataset)
os.system(f'find {source_path} -type f ! -name "*.png" -exec rm {{}} +')
# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_horses))} images of horses.")
print(f"There are {len(os.listdir(source_path_humans))} images of humans.")

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
# Load the first example of a happy face
sample_image = load_img(f"{os.path.join(source_path_horses, os.listdir(source_path_horses)[0])}")
# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

# Define the training and validation base directories
train_dir = 'horse-or-human/training'
validation_dir = 'horse-or-human/validation'
# Directory with training horse pictures
train_horses_dir = os.path.join(train_dir, 'horses')
# Directory with training humans pictures
train_humans_dir = os.path.join(train_dir, 'humans')
# Directory with validation horse pictures
validation_horses_dir = os.path.join(validation_dir, 'horses')
# Directory with validation human pictures
validation_humans_dir = os.path.join(validation_dir, 'humans')
# Check the number of images for each class and set
print(f"There are {len(os.listdir(train_horses_dir))} images of horses for training.\n")
print(f"There are {len(os.listdir(train_humans_dir))} images of humans for training.\n")
print(f"There are {len(os.listdir(validation_horses_dir))} images of horses for validation.\n")
print(f"There are {len(os.listdir(validation_humans_dir))} images of humans for validation.\n")


print("Sample horse image:")
plt.imshow(load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}"))
plt.show()
print("\nSample human image:")
plt.imshow(load_img(f"{os.path.join(train_humans_dir, os.listdir(train_humans_dir)[0])}"))
plt.show()
