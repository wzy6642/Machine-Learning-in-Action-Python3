# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:20:23 2018

@author: wzy
"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：加载数据

Parameters:
    filename - 文件名
    
Returns:
    xArr - x数据集
    yArr - y数据集

Modify:
    2018-07-30
"""
def loadDataSet(filename):
    # 计算特征个数，由于最后一列为y值所以减一
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


"""
函数说明：岭回归

Parameters:
    xMat - x数据集
    yMat - y数据集
    lam - 缩减系数
    
Returns:
    ws - 回归系数

Modify:
    2018-07-30
"""
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    demon = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 求矩阵的行列式
    if np.linalg.det(demon) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    # .I求逆矩阵
    ws = (demon.I) * (xMat.T) * yMat
    return ws


"""
函数说明：岭回归测试

Parameters:
    xArr - x数据集
    yArr - y数据集
    
Returns:
    wMat - 回归系数矩阵

Modify:
    2018-07-30
"""
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化
    # 行与行操作，求均值
    yMean = np.mean(yMat, axis=0)
    # 数据减去均值
    yMat = yMat - yMean
    # 行与行操作，求均值
    xMeans = np.mean(xMat, axis=0)
    # 行与行操作，求方差
    xVar = np.var(xMat, axis=0)
    # 数据减去均值除以方差实现标准化
    xMat = (xMat - xMeans) / xVar
    # 30个不同的lamda测试
    numTestPts = 30
    # 初始化回归系数矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 改变lamda计算回归系数
    for i in range(numTestPts):
        # lamda以e的指数变化，最初是一个非常小的数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        # 计算回归系数矩阵
        wMat[i, :] = ws.T
    return wMat


"""
函数说明：绘制岭回归系数矩阵

Parameters:
    None
    
Returns:
    None
Modify:
    2018-07-30
"""
def plotwMat():
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lamda)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lamda)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotwMat()
    