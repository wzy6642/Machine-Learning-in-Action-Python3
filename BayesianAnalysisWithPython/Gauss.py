# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:34:33 2018
python高斯分布概率密度函数

@author: wzy
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Mu可以从-1、0、1三个值中取值
mu_params = [-1, 0, 1]
# delta可以从0.5、1、1.5三个值中取值
sd_params = [0.5, 1, 1.5]
# x为从-7到7等间隔取100个点
x = np.linspace(-7, 7, 100)
# 把画布分割为3行3列
#  x- or y-axis will be shared among all subplots
# 返回figure对象、 Axes object or array of Axes objects.
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True)
for i in range(len(mu_params)):
    for j in range(len(sd_params)):
        # 选定Mu值和delta值
        mu = mu_params[i]
        sd = sd_params[j]
        # 定义一个正态分布，期望是mu，标准差是sd
        # .pdf(x)标准正态分布曲线
        y = stats.norm(mu, sd).pdf(x)
        # 绘制图形
        ax[i, j].plot(x, y)
        # 绘制图例alpha=0 表示 100% 透明
        ax[i, j].plot(0, 0, label="$\\mu$ = {:3.2f}\n$\\sigma$ = {:3.2f}".format (mu, sd), alpha=0)
        # 设置图例大小
        ax[i, j].legend(fontsize=12)
# 标注x轴
ax[2, 1].set_xlabel('$x$', fontsize=16)
# 标注y轴
ax[1, 0].set_ylabel('$pdf(x)$', fontsize=16)
# tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area. 
plt.tight_layout()
# 显示
plt.show()