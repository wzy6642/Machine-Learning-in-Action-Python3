# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:28:06 2018

@author: wzy
"""
import FP_Growth

# 导入数据
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
# 数据初始化为字典
initSet = FP_Growth.createInitSet(parsedDat)
# 创建FP树
myFPtree, myHeaderTab = FP_Growth.createTree(initSet, 100000)
myFreqList = []
# 查找频繁项集，查找至少被10万人浏览过的新闻报道
FP_Growth.mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
# 打印频繁项集（哪些报道浏览量达到支持度）
print(myFreqList)
