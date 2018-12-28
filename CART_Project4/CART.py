# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:21:14 2018

@author: wzy
"""
import matplotlib.pyplot as plt
import numpy as np
import types

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
    # plt.show()
    
    
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
        预剪枝
        
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
函数说明：判断测试输入变量是否是一颗树
        树是通过字典存储的
        
Parameters:
    obj - 测试对象

Returns:
    是否是一颗树

Modify:
    2018-08-01
"""
def isTree(obj):
    return (type(obj).__name__ == 'dict')


"""
函数说明：对树进行塌陷处理（即返回树平均值）
        
Parameters:
    tree - 树

Returns:
    树的平均值

Modify:
    2018-08-01
"""
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


"""
函数说明：后剪枝
        
Parameters:
    tree - 树
    testData - 测试集

Returns:
    树

Modify:
    2018-08-01
"""
def prune(tree, testData):
    # 如果测试集为空，则对树进行塌陷处理
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 如果有左子树或者右子树，则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 处理左子树（剪枝）
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 处理右子树（剪枝）
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 如果当前节点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, 1] - tree['right'], 2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差，则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


"""
函数说明：简单线性回归
        
Parameters:
    dataSet - 数据集

Returns:
    ws - 最佳回归系数
    X - 特征矩阵
    Y - label列向量
    
Modify:
    2018-08-02
"""
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    # 保存特征矩阵X的第一列全为1
    X[:, 1:n] = dataSet[:, 0:n-1]
    # 保存label列向量
    Y = dataSet[:, -1]
    # 简单线性回归
    xTx = X.T * X
    # 奇异矩阵不可以求逆
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannont do inverse,\n\
                        try increasing the second value of ops')
    # 求解回归系数
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
    

"""
函数说明：返回数据集的回归系数
        
Parameters:
    dataSet - 数据集

Returns:
    ws - 最佳回归系数
    
Modify:
    2018-08-02
"""
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


"""
函数说明：计算误差
        
Parameters:
    dataSet - 数据集

Returns:
    误差值
    
Modify:
    2018-08-02
"""
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # 求差值的平方和
    return sum(np.power(Y - yHat, 2))
    

if __name__ == '__main__':
    train_filename = 'exp2.txt'
    train_Data = loadDataSet(train_filename)
    dataMat = np.mat(train_Data)
    Tree = createTree(dataMat, modelLeaf, modelErr, (1, 10))
    print(Tree)
    plotDataSet(train_filename)
    # 绘制分段回归直线
    x1 = np.linspace(0, Tree['spVal'])
    plt.plot(x1, float(Tree['right'][1])*x1 + float(Tree['right'][0]), 'r--')
    x2 = np.linspace(Tree['spVal'], 1)
    plt.plot(x2, float(Tree['left'][1])*x2 + float(Tree['left'][0]), 'r--')
    # 显示
    plt.show()
    
    