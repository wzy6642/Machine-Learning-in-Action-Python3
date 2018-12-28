# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:34:57 2018

@author: wzy
"""
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：加载数据

Parameters:
    fileName - 文件名
    
Returns:
    dataMat - 数据矩阵

Modify:
    2018-08-01
"""
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 转换为float类型
        # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


"""
函数说明：根据特征切分数据集合

Parameters:
    dataSet - 数据集合
    feature - 带切分的特征
    value - 该特征的值
    
Returns:
    mat0 - 切分的数据集合0
    mat1 - 切分的数据集合1

Modify:
    2018-08-01
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


"""
函数说明：生成叶结点

Parameters:
    dataSet - 数据集合
    
Returns:
    目标变量均值

Modify:
    2018-08-01
"""
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


"""
函数说明：误差估计函数

Parameters:
    dataSet - 数据集合
    
Returns:
    目标变量的总方差

Modify:
    2018-08-01
"""
def regErr(dataSet):
    # var表示方差，即各项-均值的平方求和后再除以N
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


"""
函数说明：找到数据的最佳二元切分方式函数

Parameters:
    dataSet - 数据集合
    leafType - 生成叶结点的函数
    errType - 误差估计函数
    ops - 用户定义的参数构成的元组

Returns:
    bestIndex - 最佳切分特征
    bestValue - 最佳特征值

Modify:
    2018-08-01
"""
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # tolS：允许的误差下降值
    tolS = ops[0]
    # tolN：切分的最小样本数
    tolN = ops[1]
    # 如果当前所有值相等，则退出（根据set的特性只保留不重复的元素）
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征，计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差，最佳特征切分的索引值，最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0 
    # 遍历所有特征
    for featIndex in range(n-1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN，则退出剪枝操作
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算误差估计,寻找newS的最小值
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                # 特征索引
                bestIndex = featIndex
                # 分割标准
                bestValue = splitVal
                # 更新目标函数的最小值
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue


"""
函数说明：树构建函数

Parameters:
    dataSet - 数据集合
    leafType - 生成叶结点的函数
    errType - 误差估计函数
    ops - 用户定义的参数构成的元组

Returns:
    retTree - 构建的回归树

Modify:
    2018-08-01
"""
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果没有特征，则返回特征值
    if feat == None:
        return val
    # 回归树
    retTree = {}
    # 分割特征索引
    retTree['spInd'] = feat
    # 分割标准
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树 递归
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


"""
函数说明：绘制数据集

Parameters:
    fileName - 文件名
    
Returns:
    None

Modify:
    2018-08-01
"""
def plotDataSet(filename):
    dataMat = loadDataSet(filename)
    n = len(dataMat)
    xcord = []
    ycord = []
    # 样本点
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    # filename = 'ex00.txt'
    # plotDataSet(filename)
    myData = loadDataSet('ex2.txt')
    myMat = np.mat(myData)
    print(createTree(myMat))
    # print(feat)
    # print(val)
    