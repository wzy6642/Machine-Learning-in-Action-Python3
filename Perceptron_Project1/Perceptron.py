# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:31:10 2018

@author: wzy
"""
# 二维数据集决策边界可视化
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
类说明：构建感知机

Parameters:
    eta - 学习率(0,1]
    n_iter - 迭代次数
    w_ - 训练后的权重数组
    errors_ - 每轮训练后的误差
    
Returns:
    None

Modify:
    2018-08-28
"""
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    """
    Fit training data
    Parameters
    X: {array-like}, shape=[n_samples, n_features]
       Trainig vectors, where n_samples is the number of samples and n_features is the number of features.
    y: array-like, shape = [n_samples]
       Target values.
    Returns
    self: object
    """
    def fit(self, X, y):
        # self.w_中的权值初始化为一个零向量R(m+1),其中m是数据集中维度(特征)的数量
        # 我们在此基础上增加一个0权重列(也就是阈值)
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 每轮中错分类样本的数量
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        # 计算X和w_的点积 
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


"""
函数说明：导入数据集

Parameters:
    None
    
Returns:
    X - 特征矩阵
    y - label列向量

Modify:
    2018-08-28
"""
def DataSet():
    # 使用pandas库直接从UCI机器学习库中将鸢尾花数据集转换为DataFrame对象并加载到内存中
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # 使用tail方法显示数据最后5行以保证数据正确加载
    df.tail()
    # 提取前100个类标，50个山鸢尾类标，50个变色鸢尾类标
    # iloc works on the positions in the index (so it only takes integers).
    y = df.iloc[0:100, 4].values
    # -1代表山鸢尾 1代表变色鸢尾，将label存到y中
    # np.where用法相当于C语言的 ? : 
    y = np.where(y == 'Iris-setosa', -1, 1)
    # 提取特征0和特征1
    X = df.iloc[0:100, [0, 2]].values
    # 绘制散点图
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # 花瓣长度
    plt.xlabel('petal length')
    # 萼片长度
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()
    # X为特征矩阵，y为label列向量（-1，1）
    return X, y


"""
函数说明：绘制迭代次数与误分点个数之间的关系

Parameters:
    None
    
Returns:
    None

Modify:
    2018-08-28
"""
def NumOfErrors():
    # 导入数据
    X, y = DataSet()
    # 实例化感知机
    ppn = Perceptron(eta=0.1, n_iter=10)
    # 训练模型
    ppn.fit(X, y)
    # 绘制迭代次数与误分点个数之间的关系
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    

"""
函数说明：绘制决策区域图像

Parameters:
    X - 特征矩阵
    y - label列向量
    classifier - 分类器
    resolution - 采样间隔为0.02
    
Returns:
    None

Modify:
    2018-08-28
"""
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 散点样式
    markers = ('s', 'x', 'o', '^', 'v')
    # 颜色元组
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # np.unique该函数是去除数组中的重复数字，并进行排序之后输出。
    # ListedColormap主要用于生成非渐变的颜色映射
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 横轴范围
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # 纵轴范围
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # meshgrid函数将最大值、最小值向量生成二维数组xx1和xx2
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        

if __name__ == '__main__':
    # 导入数据
    X, y = DataSet()
    # 实例化感知机
    ppn = Perceptron(eta=0.1, n_iter=10)
    # 训练模型
    ppn.fit(X, y)
    plot_decision_regions(X, y, classifier=ppn)
    # 萼片长度
    plt.xlabel('sepal length [cm]')
    # 花瓣长度
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
    