# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:53:40 2018

@author: wzy
"""
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：将文本文档中的数据读入到python中

Parameters:
    fileName - 文件名
    
Returns:
    dataMat - 数据矩阵

Modify:
    2018-08-02
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


"""
函数说明：数据向量计算欧式距离

Parameters:
    vecA - 数据向量A
    vecB - 数据向量B
    
Returns:
    两个向量之间的欧几里德距离

Modify:
    2018-08-02
"""
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


"""
函数说明：随机初始化k个质心（质心满足数据边界之内）

Parameters:
    dataSet - 输入的数据集
    k - 选取k个质心
    
Returns:
    centroids - 返回初始化得到的k个质心向量

Modify:
    2018-08-02
"""
def randCent(dataSet, k):
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据集的每一个维度
    for j in range(n):
        # 得到该列数据的最小值,最大值
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(maxJ - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    # 返回初始化得到的k个质心向量
    return centroids


"""
函数说明：k-means聚类算法

Parameters:
    dataSet - 用于聚类的数据集
    k - 选取k个质心
    distMeas - 距离计算方法,默认欧氏距离distEclud()
    createCent - 获取k个质心的方法,默认随机获取randCent()
    
Returns:
    centroids - k个聚类的聚类结果
    clusterAssment - 聚类误差

Modify:
    2018-08-02
"""
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    # 初始化一个（m,2）全零矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不发生变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为False
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if clusterAssment[i, 0] != minIndex: 
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist**2
            # 打印k-means聚类的质心
        # print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算这些数据的均值(axis=0:求列均值)，作为该类质心向量
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment


"""
函数说明：二分k-means聚类算法

Parameters:
    dataSet - 用于聚类的数据集
    k - 选取k个质心
    distMeas - 距离计算方法,默认欧氏距离distEclud()
    
Returns:
    centList - k个聚类的聚类结果
    clusterAssment - 聚类误差

Modify:
    2018-08-03
"""
def biKmeans(dataSet, k, distMeas=distEclud):
    # 获取数据集的样本数
    m = np.shape(dataSet)[0]
    # 初始化一个元素均值0的(m, 2)矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 获取数据集每一列数据的均值，组成一个列表
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 当前聚类列表为将数据集聚为一类
    centList = [centroid0]
    # 遍历每个数据集样本
    for j in range(m):
        # 计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
    # 循环，直至二分k-Means值达到k类为止
    while (len(centList) < k):
        # 将当前最小平方误差置为正无穷
        lowerSSE = float('inf')
        # 遍历当前每个聚类
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对该类利用二分k-means算法进行划分，返回划分后的结果以及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit = np.sum(splitClustAss[:, 1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # 打印这两项误差值
            print('sseSplit = %f, and notSplit = %f' % (sseSplit, sseNotSplit))
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowerSSE:
                # 第i类作为本次划分类
                bestCentToSplit = i
                # 第i类划分后得到的两个质心向量
                bestNewCents = centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowerSSE = sseSplit + sseNotSplit
        # 数组过滤选出本次2-means聚类划分后类编号为1数据点，将这些数据点类编号变为
        # 当前类个数+1， 作为新的一个聚类
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 同理，将划分数据中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        # 连续不出现空缺
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # 打印本次执行2-means聚类算法的类
        print('the bestCentToSplit is %d' % bestCentToSplit)
        # 打印被划分的类的数据个数
        print('the len of bestClustAss is %d' % len(bestClustAss))
        # 更新质心列表中变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 添加新的类的质心向量
        centList.append(bestNewCents[1, :])
        # 更新clusterAssment列表中参与2-means聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    # 返回聚类结果
    return centList, clusterAssment


"""
函数说明：绘制数据集

Parameters:
    fileName - 文件名
    k - 选取k个质心
    
Returns:
    None

Modify:
    2018-08-01
"""
def plotDataSet(filename, k):
    # 导入数据
    datMat = np.mat(loadDataSet(filename))
    # 进行k-means算法其中k为4
    centList, clusterAssment = biKmeans(datMat, k)
    clusterAssment = clusterAssment.tolist()
    xcord = [[], [], []]
    ycord = [[], [], []]
    datMat = datMat.tolist()
    m = len(clusterAssment)
    for i in range(m):
        if int(clusterAssment[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clusterAssment[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    # 绘制质心
    for i in range(k):
        ax.scatter(centList[i].tolist()[0][0], centList[i].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    # ax.scatter(centList[0].tolist()[0][0], centList[0].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    # ax.scatter(centList[1].tolist()[0][0], centList[1].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    # ax.scatter(centList[2].tolist()[0][0], centList[2].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    
    
if __name__ == '__main__':
    datMat = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat, 3)
    plotDataSet('testSet2.txt', 3)
    