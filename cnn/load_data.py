import os

import cv2
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from neural_network_model import check_model, nn_model, check_model_new, evaluate_model_, evaluate_new_model

dataset_dir = "../../img_she_healthy/DATASET/crvx"

m_batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=m_batch_size,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

validation_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=m_batch_size,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

model = nn_model(2)
epoch = 100

history = check_model_new(model, train_generator, validation_generator, epoch, m_batch_size)
evaluate_new_model(history)
