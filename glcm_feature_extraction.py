import cv2

from calculate_glcm_algo import calculate_glcm

def glcm_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm_features = []

    features = calculate_glcm(gray)
    glcm_features.append(features)

    hasil = ""
    for name in glcm_features[0]:
        hasil += str(name) + ","

    return hasil
