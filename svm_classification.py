# importing necessary libraries
import cv2
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from calculate_glcm_algo import calculate_glcm


def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data / (10 ** c)


# loading the glcm dataset
glcm_df = pd.read_csv("glcm_cervix.csv")

# X -> features, y -> label
glcm_features = glcm_df[['contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
                         'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                         'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
                         'energy_0', 'energy_45', 'energy_90', 'energy_135',
                         'dissimilarity_0', 'dissimilarity_45', 'dissimilarity_90', 'dissimilarity_135',
                         'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135']].values

X = glcm_features

y = glcm_df['label'].values

print("X no scaling is ", X.shape)

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train)
print(y_train)

print(X_test)
print(y_test)

# print(X_train.shape)
print(y_train.shape)
# print(X_test.shape)
print(y_test.shape)

# print(type(X_train), type(y_train))
# print(type(X_test), type(y_test))

# training a linear SVM classifier 
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
print("X_TEST ", X_test.shape)

img = cv2.imread("../img_she_healthy/testing/precancer7.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

glcm_features = []

features = calculate_glcm(gray)
glcm_features.append(features)

X = decimal_scaling(glcm_features)

print("REAL X ", X.shape)

# svm_predictions = svm_model_linear.predict(X)

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test)

# creating a confusion matrix 
# cm = confusion_matrix(y_test, svm_predictions)

# print("Prediction is", svm_predictions)

print("Accuracy adalah ", accuracy * 100, "%")
# print(cm)
