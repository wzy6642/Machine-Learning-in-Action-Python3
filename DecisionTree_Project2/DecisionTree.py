# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 21:52:06 2018
数据的Labels依次是age、prescript、astigmatic、tearRate、class
年龄、症状、是否散光、眼泪数量、分类标签
@author: wzy
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import tree

if __name__ == '__main__':
    # 加载文件
    with open('lenses.txt') as fr:
        # 处理文件，去掉每行两头的空白符，以\t分隔每个数据
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_targt = []
    for each in lenses:
        # 存储Label到lenses_targt中
        lenses_targt.append([each[-1]])
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典
    for each_label in lensesLabels:
        for each in lenses:
            # index方法用于从列表中找出某个值第一个匹配项的索引位置
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印字典信息
    # print(lenses_dict)
    # 生成pandas.DataFrame用于对象的创建
    lenses_pd = pd.DataFrame(lenses_dict)
    # 打印数据
    # print(lenses_pd)
    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        # fit_transform()干了两件事：fit找到数据转换规则，并将数据标准化
        # transform()直接把转换规则拿来用,需要先进行fit
        # transform函数是一定可以替换为fit_transform函数的，fit_transform函数不能替换为transform函数
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # 打印归一化的结果
    # print(lenses_pd)
    # 创建DecisionTreeClassifier()类
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    # 使用数据构造决策树
    # fit(X,y):Build a decision tree classifier from the training set(X,y)
    # 所有的sklearn的API必须先fit
    clf = clf.fit(lenses_pd.values.tolist(), lenses_targt)
    dot_data = StringIO()
    # 绘制决策树
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=clf.classes_, filled=True, rounded=True, 
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 保存绘制好的决策树，以PDF的形式存储。
    graph.write_pdf("tree.pdf")
    #预测
    print(clf.predict([[1,1,1,0]]))      
              