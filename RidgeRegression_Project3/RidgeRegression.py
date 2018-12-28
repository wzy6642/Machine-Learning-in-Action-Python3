# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:11:07 2018

@author: wzy
"""
from sklearn import linear_model
from bs4 import BeautifulSoup

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
函数说明：使用sklearn

Parameters:
    None
    
Returns:
    None

Modify:
    2018-07-30
"""
def usesklearn():
    reg = linear_model.Ridge(alpha=.5)
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    # fit(X, y):Fit Ridge regression model
    reg.fit(lgX, lgY)
    print("%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价" % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))


if __name__ == '__main__':
    usesklearn()
    