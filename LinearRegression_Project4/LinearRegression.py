# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:17:56 2018

@author: wzy
"""
import numpy as np
from bs4 import BeautifulSoup
import random

"""
函数说明：从页面读取数据，生成retX和retY列表

Parameters:
    retX - 数据X
    retY - 数据Y
    inFile - HTML文件
    yr - 年份
    numPce - 乐高部件数目
    origPrc - 原价
    
Returns:
    None

Modify:
    2018-07-30
"""
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r='%d' % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r='%d' % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if(lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品#%d没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print('%d\t%d\t%d\t%f\t%f' % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r='%d' % i)


"""
函数说明：依次读取六种乐高套装的数据，并生成数据矩阵

Parameters:
    retX - 数据X
    retY - 数据Y
    
Returns:
    None

Modify:
    2018-07-30
"""
def setDataCollect(retX, retY):
    # 2006年的乐高8288，部件数目800，原价49.99
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)


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
    # 深层拷贝
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
    print(inMeans)
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
函数说明：计算回归系数w

Parameters:
    xArr - x数据集
    yArr - y数据集
    
Returns:
    ws - 回归系数

Modify:
    2018-07-30
"""
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 求矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    # .I求逆矩阵
    ws = (xTx.I) * (xMat.T) * yMat
    return ws


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
函数说明：使用简单的线性回归

Parameters:
    None
    
Returns:
    None

Modify:
    2018-07-30
"""
def useStandRegres():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    data_num, features_num = np.shape(lgX)
    # 第一列全为1
    lgx1 = np.mat(np.ones((data_num, features_num+1)))
    lgx1[:, 1:5] = np.mat(lgX)
    # 计算回归系数
    ws = standRegres(lgx1, lgY)
    print("%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价" % (ws[0], ws[1], ws[2], ws[3], ws[4]))
    

"""
函数说明：交叉验证岭回归

Parameters:
    xArr - x数据集
    yArr - y数据集
    numVal - 交叉验证次数
    
Returns:
    wMat - 回归系数矩阵

Modify:
    2018-07-30
"""
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        # shuffle() 方法将序列的所有元素随机排序。
        random.shuffle(indexList)
        for j in range(m):
            # 90%数据训练集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            # 10%数据测试集
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 岭回归测试
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            # 标准化
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            # 数据还原
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
    meanErrors = np.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    # 表转换
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print("%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价" % ((-1 * np.sum(np.multiply(meanX, unReg)) + np.mean(yMat)), unReg[0, 0], unReg[0, 1], unReg[0, 2], unReg[0, 3]))
    
 
if __name__ == '__main__':
    # useStandRegres()
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    print(ridgeTest(lgX, lgY))
    crossValidation(lgX, lgY)
    