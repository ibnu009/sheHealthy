import cv2
import numpy as np

from calculate_glcm_algo import calculate_glcm
from neural_network_model import precision, recall
import keras


def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data / (10 ** c)


def cnn_classification():
    img = cv2.imread("./testing/precancer2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_features = []

    features = calculate_glcm(gray)
    glcm_features.append(features)

    X = decimal_scaling(glcm_features)

    print("Properties adalah ", glcm_features)
    print("X adalah ", X.shape)
    print("Image adalah ", img.shape)
    print("Grey adalah ", gray.shape)

    reconstructed_model = keras.models.load_model("my_h5_model.h5",
                                                  custom_objects={'precision': precision, 'recall': recall})
    pred = reconstructed_model.predict(X)

    print("", pred.argmax(axis=1))

    hasil = ""
    for name in glcm_features[0]:
        hasil += str(name) + ","

    if pred.argmax(axis=1)[0] == 1:
        print("Hasil prediksi adalah NORMAL")
        hasil += "NORMAL"
    else:
        hasil += "PRECANCER"
        print("Hasil prediksi adalah PRECANCER")

    # cb = [glcm_features, ]
    #
    # print("HAHA", cb)

    return hasil
