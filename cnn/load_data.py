import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

from neural_network_model import nn_model, check_model_new, evaluate_new_model

# Ini path ke datasets mu
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
    target_size=(64, 64), #Ini nanti diubah ke 224 x 224
    batch_size=m_batch_size,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

model = nn_model()
epoch = 50
history = check_model_new(model, train_generator, validation_generator, epoch, m_batch_size)

prediction_generator = model.predict_generator(train_generator)

predicted_labels = [np.argmax(prediction) for prediction in prediction_generator]

confusion_mat = confusion_matrix(train_generator.classes, predicted_labels)

print(confusion_mat)

evaluate_new_model(history)
