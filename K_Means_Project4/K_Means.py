# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:55:01 2018

@author: wzy
"""
# 聚类数据生成器
from sklearn.datasets import make_blobs
# KMeans算法使用
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 生成聚类数据 150个样本，每个样本两个特征，一共聚为3簇，簇内标准差为0.5，所有样本数据随机排序，采用随机数种子
    x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    # 绘制样本点的散点图
    plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
    # 聚为3个簇，从训练数据用k-means++寻找质心，初始样本中心的个数为10个，最大迭代次数为300，SSE为10^(-4)
    km = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0)
    # 用K-means计算并且将X作为测试集分簇
    y_km = km.fit_predict(x)
    # 绘制不同簇的点
    plt.scatter(x[y_km==0, 0], x[y_km==0, 1], s=50, c='orange', marker='o', label='cluster 1')
    plt.scatter(x[y_km==1, 0], x[y_km==1, 1], s=50, c='green', marker='s', label='cluster 2')
    plt.scatter(x[y_km==2, 0], x[y_km==2, 1], s=50, c='blue', marker='^', label='cluster 3')
    # 绘制簇的中心点
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker="*", c="red", label="cluster center")
    plt.legend()
    plt.grid()
    plt.show()
