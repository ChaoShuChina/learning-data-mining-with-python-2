__author__ = 'chao-shu'
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=14)
estimator = KNeighborsClassifier(n_neighbors=10)
estimator.fit(x_train, y_train)

y_predicted = estimator.predict(x_test)
accuracy = np.mean(y_test == y_predicted) * 100

scores = cross_val_score(estimator, x, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print(accuracy)
print(average_accuracy)

avg_scores = []
all_scores = []
parameter_value = list(range(1, 21))
for n_neighbor in parameter_value:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbor)
    scores = cross_val_score(estimator, x, y)
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

plt.plot(parameter_value, avg_scores, '-o')

print(1)