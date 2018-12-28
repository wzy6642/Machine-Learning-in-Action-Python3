# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:04:53 2018

@author: wzy
"""
import numpy as np
from scipy.stats import binom

# 采集数据，1表示正面，0表示反面
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])


"""
函数说明：EM算法单次迭代

Parameters:
    priors - [theta_A, theta_B]A、B各自为正面向上的概率
    observations - 数据矩阵
    
Returns:
    [new_theta_A, new_theta_B] - 新的A、B各自为正面向上的概率

Modify:
    2018-08-29
"""
def em_single(priors, observations):
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]
    # E-step
    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads
        # 二项分布求解公式
        contribution_A = binom.pmf(num_heads, len_observation, theta_A)
        contribution_B = binom.pmf(num_heads, len_observation, theta_B)
        # 计算两个概率
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        # 估计数据中A、B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails
    # M-step
    # 计算新模型的参数
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A, new_theta_B]


"""
函数说明：EM算法主循环

Parameters:
    prior - [theta_A, theta_B]A、B各自为正面向上的概率初值
    observations - 数据矩阵
    tol - 迭代结束的阈值
    iterations - 最大迭代次数
    
Returns:
    [new_prior, iteration] - 局部最优的模型参数，最新的A、B各自为正面向上的概率以及收敛时迭代次数

Modify:
    2018-08-29
"""
def em(observations, prior, tol=1e-6, iterations=10000):
    iteration = 0
    while iteration < iterations:
        new_prior = em_single(prior, observations)
        delta_change = np.abs(prior[0] - new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration += 1
    return [new_prior, iteration]
        

if __name__ == '__main__':
    print(em(observations, [0.99999, 0.00001]))
    