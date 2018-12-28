# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:57:55 2018

@author: wzy
"""
import matplotlib
# 设置后端TkAgg
matplotlib.use('TkAgg')
# 将TkAgg和matplotlib连接起来
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# 图框
from matplotlib.figure import Figure
import numpy as np
# 绘图工具包
import tkinter as tk
# 调用CART回归树中的代码
import CART

"""
函数说明：绘制原始数据的散点图以及拟合数据的曲线图
        
Parameters:
    tolS - 允许的误差下降值
    tolN - 切分的最小样本数

Returns:
    None
    
Modify:
    2018-08-02
"""
def reDraw(tolS, tolN):
    # 清空画布
    reDraw.f.clf()
    # 只有一个格子用于填充图形
    reDraw.a = reDraw.f.add_subplot(111)
    # 检查复选框是否选中
    # 选中调用模型树进行回归
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = CART.createTree(reDraw.rawDat, CART.modelLeaf, CART.modelErr, (tolS, tolN))
        yHat = CART.createForeCast(myTree, reDraw.testDat, CART.modelTreeEval)
    # 没选中调用回归树
    else:
        myTree = CART.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = CART.createForeCast(myTree, reDraw.testDat)
    # 绘制真实值的散点图
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    # 绘制预测值曲线
    reDraw.a.plot(reDraw.testDat, yHat, 'b', linewidth=2.0)
    # 画布显示
    reDraw.canvas.show()


"""
函数说明：获取文本框输入值
        
Parameters:
    None

Returns:
    None
    
Modify:
    2018-08-02
"""
def getInputs():
    # 期望输入为整数
    try: 
        tolN = int(tolNentry.get())
    # 清除错误用默认替换
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    # 期望输入为浮点数
    try: 
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


"""
函数说明：根据文本框输入参数绘图
        
Parameters:
    None

Returns:
    None
    
Modify:
    2018-08-02
"""
def drawNewTree():
    # 从文本框中获取参数
    tolN, tolS = getInputs()
    # 绘制图
    reDraw(tolS, tolN)


# 创建窗口
root = tk.Tk()
reDraw.f = Figure(figsize=(5,4), dpi=100)
# matplotlib的后端操作
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# 添加文字标签 网格布局 columnspan和rowspan的值告诉布局是否允许跨列或跨行
tk.Label(root, text="tolN").grid(row=1, column=0)
# 添加文本输入框
tolNentry = tk.Entry(root)
tolNentry.grid(row=1, column=1)
# 默认填入10
tolNentry.insert(0, '10')
tk.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
# 添加按键控件，按下连接到drawNewTree函数
tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
# 按钮整数数值用来读取Checkbutton状态
chkBtnVar = tk.IntVar()
# 添加复选框
chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
# 读入数据
reDraw.rawDat = np.mat(CART.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
# 监听事件
root.mainloop()
