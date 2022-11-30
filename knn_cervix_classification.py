import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

glcm_df = pd.read_csv("glcm_cervix.csv")

glcm_df.head()

X = glcm_df.drop(columns=['label'])
y = glcm_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

print(X_train)
print(y_train)

print(X_test)
print(y_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # avoid data leakage

print(X_train)
print(X_test.dtype)

from math import sqrt
class KNN():
    def __init__(self, k):
        self.k = k
        print(self.k)

    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train

    def calculate_euclidean(self, sample1, sample2):
        distance = 0.0
        for i in range(len(sample1)):
            distance += (sample1[i] - sample2[i]) ** 2  # Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
        return sqrt(distance)

    def nearest_neighbors(self, test_sample):
        distances = []  # calculate distances from a test sample to every sample in a training set
        for i in range(len(self.x_train)):
            distances.append((self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample)))
        distances.sort(key=lambda x: x[1])  # sort in ascending order, based on a distance value
        neighbors = []
        for i in range(self.k):  # get first k samples
            neighbors.append(distances[i][0])
        return neighbors

    def predict(self, test_set):
        predictions = []
        for test_sample in test_set:
            neighbors = self.nearest_neighbors(test_sample)
            labels = [sample for sample in neighbors]
            prediction = max(labels, key=labels.count)
            predictions.append(prediction)
            return predictions

model = KNN(5) #our model
model.fit(X_train, y_train)

classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)  #The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
predictions = model.predict(X_test)  # our model's predictions
print(predictions)

cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

cm = confusion_matrix(y_test, predictions)  # our model
print(cm)
acs = accuracy_score(y_test, predictions)
print(acs)


