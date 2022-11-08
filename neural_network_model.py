from keras.models import Sequential
from keras.layers import Dense, Activation

import keras
from keras import backend as K
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

# --------------------- create custom metric evaluation ---------------------
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# --------------------------- create model -------------------------------
def nn_model(max_len):
    model = Sequential()
    model.add(Dense(32,
                    activation="elu",
                    input_shape=(max_len,)))
    model.add(Dense(1024, activation="elu"))
    model.add(Dense(512, activation="elu"))
    model.add(Dense(256, activation="elu"))
    model.add(Dense(128, activation="elu"))
    model.add(Dense(2))
    model.add(Activation("sigmoid"))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall])

    return model

# ------------------------- check model -----------------------------

checkpoint_path = "best_model.h5"

mc = ModelCheckpoint(filepath=checkpoint_path,
                     monitor='accuracy',
                     verbose=1,
                     save_best_only=True)

es = EarlyStopping(monitor='accuracy',
                   min_delta=0.01,
                   patience=5,
                   verbose=1)

cb = [mc, es]


def check_model(model_, x, y, x_val, y_val, epochs_, batch_size_):
    hist = model_.fit(x,
                      y,
                      epochs=epochs_,
                      batch_size=batch_size_,
                      validation_data=(x_val, y_val),
                      # callbacks=cb
                      )

    model_.save("my_h5_model.h5")

    return hist


def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'],
             ['loss', 'val_loss'],
             ['precision', 'val_precision'],
             ['recall', 'val_recall']]
    for name in names:
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        # plt.show()
