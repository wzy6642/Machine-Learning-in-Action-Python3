#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按照y_train着色
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_mew.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
# 模型评价
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
