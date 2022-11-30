import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from confusion_matrix import plot_confusion_matrix
from neural_network_model import check_model, nn_model, evaluate_model_

glcm_df = pd.read_csv("glcm_cervix.csv")

glcm_df.head()

label_distr = glcm_df['label'].value_counts()

label_name = ['precancer', 'normal']

plt.figure(figsize=(20, 10))

my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')

p = plt.gcf()
p.gca().add_artist(my_circle)

print(label_distr)


def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data / (10 ** c)


X = decimal_scaling(
    glcm_df[['contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
             'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
             'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
             'energy_0', 'energy_45', 'energy_90', 'energy_135']].values
)

le = LabelEncoder()
le.fit(glcm_df["label"].values)

print(" categorical label : \n", le.classes_)

print("X scaling is ", X.shape)

Y = le.transform(glcm_df['label'].values)
Y = to_categorical(Y)

print("\n\n one hot encoding for sample 0 : \n", Y[0])

X_train, X_test, y_train, y_test = \
    train_test_split(X,
                     Y,
                     random_state=42)

print("Dimensi data :\n")
print("X train \t X test \t Y train \t Y test")
print("%s \t %s \t %s \t %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

max_len = X_train.shape[1]

print("Maks len adalah ", max_len)

EPOCHS = 100
BATCH_SIZE = 32

model = nn_model(max_len)
history = check_model(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE)
evaluate_model_(history)

# predict test data
y_pred = model.predict(X_test)

# print("Hasil predict raw adalah ", y_pred)
print("Hasil predict adalah ", y_pred.argmax(axis=1))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,
                      classes=['precancer', 'normal'],
                      normalize=False,
                      title='Confusion matrix, with normalization')

print(classification_report(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=['precancer', 'normal']))
