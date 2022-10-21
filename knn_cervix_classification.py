import pandas as pd

dataset = pd.read_csv('glcm_cervix.csv')
dataset.groupby('label').size()

normal = dataset[dataset['label'] == 'normal']
preCancer = dataset[dataset['label'] == 'precancer']

print(normal.describe())
