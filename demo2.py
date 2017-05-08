__author__ = 'chao-shu'
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import os

data_filename = "/home/chao-shu/data/ionosphere.data"
x = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        x[i] = data
        y[i] = row[-1] == 'g'

x_broken = np.array(x)
x_broken[:,::2] /= 10

estimator = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(estimator, x, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print(average_accuracy)

scores = cross_val_score(estimator, x_broken, y, scoring='accuracy')
average_accuracy1 = np.mean(scores) * 100
print(average_accuracy1)


x_transformed = MinMaxScaler().fit_transform(x_broken)
# print(x_transformed)
transformed_scores = cross_val_score(estimator, x_transformed, y, scoring='accuracy')
average_accuracy2 = np.mean(transformed_scores) * 100
print(average_accuracy2)