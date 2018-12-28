# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:02:48 2018
书上给出的yahooAPI的baseurl已经改变，github上有oauth2供python使用，
但是yahoo的BOOS GEO好像OAuth2验证出了问题，虽然写了新的placeFinder调用api的代码，
仍然会有403错误。
好在随书代码中已经给出place.txt，所以直接调用，这里略过获取数据的步骤。

@author: wzy
"""
import urllib
import json
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import KMeans
import math as math
"""
函数说明：利用地名、城市获取位置处经纬度

Parameters:
    stAddress - 地名
    city - 城市
    
Returns:
    None

Modify:
    2018-08-03
"""
def geoGrab(stAddress, city):
    # 获取经纬度网址
    apiStem = "http://where.yahooapis.com/geocode?"
    # 初始化一个字典，存储相关参数
    params = {}
    # 返回类型为json
    params['flags'] = 'J'
    # 参数appid
    params['appid'] = 'ppp68N8t'
    # 参数地址位置信息
    params['location'] = ('%s %s' % (stAddress, city))
    # 利用urlencode函数将字典转为URL可以传递的字符串格式
    url_params = urllib.parse.urlencode(params)
    # 组成完整的URL地址api
    yahooApi = apiStem + url_params
    # 打印该URL地址
    print('%s' % yahooApi)
    # 打开URL，返回JSON格式数据
    c = urllib.request.urlopen(yahooApi)
    # 返回JSON解析后的数据字典
    return json.load(c.read())


"""
函数说明：具体文本数据批量地址经纬度获取

Parameters:
    fileName - 文件名称
    
Returns:
    None

Modify:
    2018-08-03
"""
def massPlaceFind(fileName):
    # "wb+" 以二进制写方式打开,可以读\写文件,如果文件不存在,创建该文件.如果文件已存在,先清空,再打开文件
    # 以写方式打开,只能写文件,如果文件不存在,创建该文件如果文件已存在,先清空,再打开文件
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        # 获取该地址的经纬度
        retDict = geoGrab(lineArr[1], lineArr[2])
        # 获取到相应的经纬度
        if retDict['ResultSet']['Error'] == 0:
            # 从字典中获取经度
            lat = float(retDict['ResultSet']['Results'][0]['latitute'])
            # 从字典中获取维度
            lng = float(retDict['ResultSet']['Results'][0]['longitute'])
            # 打印地名及对应的经纬度信息
            print('%s\t%f\t%f' % (lineArr[0], lat, lng))
            # 保存入文件
            fw.write('%s\t%f\t%f' % (line, lat, lng))
        else:
            print('error fetching')
        # 为防止频繁调用API，造成请求被封，使函数调用延迟一秒
        sleep(1)
    # 文本写入关闭
    fw.close()


"""
函数说明：球面距离计算

Parameters:
    vecA - 数据向量A
    vecB - 数据向量B
    
Returns:
    球面距离

Modify:
    2018-08-03
"""
def distSLC(vecA, vecB):
    a = math.sin(vecA[0, 1] * np.pi / 180) * math.sin(vecB[0, 1] * np.pi / 180)
    b = math.cos(vecA[0, 1] * np.pi / 180) * math.cos(vecB[0, 1] * np.pi / 180) * math.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return math.acos(a + b) * 6371.0


"""
函数说明：k-means聚类

Parameters:
    numClust - 聚类个数
    
Returns:
    None

Modify:
    2018-08-03
"""
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    # 利用2-means聚类算法聚类
    myCentroids, clustAssing = KMeans.biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], \
                    ptsInCurrCluster[:, 1].flatten().A[0], \
                    marker = markerStyle, s=90)
    for i in range(numClust):
        ax1.scatter(myCentroids[i].tolist()[0][0], myCentroids[i].tolist()[0][1], s=300, c='k', marker='+', alpha=.5)
    plt.show()
        
        
if __name__ == '__main__':
    # 不能通过URL访问了
    # massPlaceFind('portlandClubs.txt')
    clusterClubs()
    