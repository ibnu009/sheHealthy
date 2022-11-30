import cv2
import numpy as np

from calculate_glcm_algo import calculate_glcm


def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data / (10 ** c)


def glcm_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm_features = []

    features = calculate_glcm(gray)
    glcm_features.append(features)

    hasil = ""
    for name in glcm_features[0]:
        hasil += str(name) + ","

    return hasil
