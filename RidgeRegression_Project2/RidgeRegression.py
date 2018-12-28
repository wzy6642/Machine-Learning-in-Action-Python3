# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:37:02 2018

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
函数说明：数据标准化

Parameters:
    xMat - x数据集
    yMat - y数据集
    
Returns:
    inxMat - 标准化后的x数据集
    inyMat - 标准化后的y数据集

Modify:
    2018-07-30
"""
def regularize(xMat, yMat):
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    # 求yMat的均值
    yMean = np.mean(yMat, 0)
    # 计算yMat每一个值与yMean的差值
    inyMat = yMat - yMean
    # 求inxMat每一列的均值
    inMeans = np.mean(inxMat, 0)
    # 求inxMat每一列的方差即（各项-均值的平方求和）后再除以N
    inVar = np.var(inxMat, 0)
    # 数据减去均值处以方差实现标准化
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat


"""
函数说明：计算平方误差

Parameters:
    yArr - 预测值
    yHatArr - 真实值
    
Returns:
    平方误差

Modify:
    2018-07-30
"""
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


"""
函数说明：前向逐步线性回归

Parameters:
    xArr - x输入数据
    yArr - y输入数据
    eps - 每次迭代需要调整的步长
    numIt - 迭代次数
    
Returns:
    returnMat - numIt次迭代的回归系数矩阵

Modify:
    2018-07-30
"""
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    # 使xMat和yMat的行数一致
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 标准化
    xMat, yMat = regularize(xMat, yMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                # 微调回归系数
                wsTest[j] += eps * sign
                # 计算预测值
                yTest = xMat * wsTest
                # 计算平方误差
                rssE = rssError(yMat.A, yTest.A)
                # 如果误差更小则更新当前的最佳回归系数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # 记录numIt次迭代的回归系数矩阵
        returnMat[i, :] = ws.T
    return returnMat


"""
函数说明：绘制岭回归系数矩阵

Parameters:
    None
    
Returns:
    None
Modify:
    2018-07-30
"""
def plotstageWiseMat():
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归：迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotstageWiseMat()
    