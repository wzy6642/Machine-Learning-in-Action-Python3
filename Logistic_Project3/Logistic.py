# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:53:31 2018

@author: wzy
"""
from sklearn.linear_model import LogisticRegression

"""
函数说明：使用Sklearn构建Logistic回归分类器

Parameters:
    None
    
Returns:
    None

Modify:
    2018-07-22
"""
def colicSklearn():
    # 打开训练集
    frTrain = open('horseColicTraining.txt')
    # 打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    # fit(X,y) Fit the model according to the given training data
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    # score(X,y) Returns the mean accuracy on the given test data and labels
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print("正确率为：%f%%" % test_accurcy)
    

if __name__ == '__main__':
    colicSklearn()