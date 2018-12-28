# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:50:58 2018

@author: wzy
"""
import numpy as np
import matplotlib.pyplot as plt

"""
函数说明：解析文本数据

Parameters:
    filename - 文件名
    delim - 每一行不同特征数据之间的分隔方式，默认是tab键‘\t’
    
Returns:
    j将float型数据值列表转化为矩阵返回

Modify:
    2018-08-07
"""
def loadDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


"""
函数说明：PCA特征维度压缩函数

Parameters:
    dataMat - 数据集数据
    topNfeat - 需要保留的特征维度，即要压缩成的维度数，默认4096
    
Returns:
    lowDDataMat - 压缩后的数据矩阵
    reconMat - 压缩后的数据矩阵反构出原始数据矩阵

Modify:
    2018-08-07
"""
def pca(dataMat, topNfeat=4096):
    # 求矩阵每一列的均值
    meanVals = np.mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
    # cov(x, 0) = cov(x)除数是n-1(n为样本个数)
    # cov(x, 1)除数是n
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[: -(topNfeat+1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    # 此处用转置和逆的结果一样redEigVects.I
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print(reconMat)
    # 返回压缩后的数据矩阵及该矩阵反构出原始数据矩阵
    return lowDDataMat, reconMat


"""
函数说明：缺失值处理函数

Parameters:
    None
    
Returns:
    datMat - 处理后的数据集

Modify:
    2018-08-07
"""
def replaceNaNWithMean():
    # 解析数据
    datMat = loadDataSet('secom.data', ' ')
    # 获取特征维度
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 利用该维度所有非NaN特征求取均值
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, 1].A))[0], i])
        # 若均值也是NaN则用0代替
        if (np.isnan(meanVal)):
            meanVal = 0
        # 将该维度中所有NaN特征全部用均值替换
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    dataMat = replaceNaNWithMean()
    # lowDmat, reconMat = pca(dataMat, topNfeat=20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=90, c='red')
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 特征数
    i = 6
    ax.scatter(range(i), eigVals[:i], marker='^', s=50, c='red')
    ax.plot(range(i), eigVals[:i])
    lowDmat, reconMat = pca(dataMat, topNfeat=i)
    # 提取的6个特征
    print(lowDmat)