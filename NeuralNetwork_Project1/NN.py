# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:25:47 2018
神经网络通过调整隐藏节点数、世代数以及学习率改善训练结果
hidden_nodes、epochs、learning_rate

@author: wzy
"""
import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot as plt

"""
类说明：构建神经网络

Parameters:
    None
    
Returns:
    None

Modify:
    2018-08-13
"""
class neuralNetwork:
    # 神经网络的构造函数
    # 初始化函数——设定输入层节点、隐藏层节点和输出层节点的数量。
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入节点、隐藏节点、输出节点、学习率
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # 通过wih和who链接权重矩阵
        # weights inside the arrays are w_i_j,where link is from node i to node j in the next layer
        # 使用正态概率分布采样权重，平均值为0.0，标准方差为节点传入链接数目的开方，即1/sqrt(传入链接数目)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # 神经网络的训练函数
    # 训练——学习给定训练集样本后，优化权重。
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    
    # 神经网络的查询函数
    # 查询——给定输入，从输出节点给出答案。
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
   

if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.2
    # 进行两轮训练
    epochs = 5
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 导入训练数据
    training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # 数据可视化处理
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            # 将数据进行归一化处理，落在区间[0.01, 1.0]内
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # np.asfarray()将文本字符串转换成实数，并创建这些数字的数组
            # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
            # output nodes is 10(example)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            # cmap='Greys'灰度图
            # plt.imshow(image_array, cmap='Greys', interpolation='None')
            n.train(inputs, targets)
    # 导入测试数据
    test_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go through all the records in the test data set
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, 'correct label')
        # 将数据进行归一化处理，落在区间[0.01, 1.0]内
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 查询神经网络
        outputs = n.query(inputs)
        # np.argmax()发现数组中的最大值，并告诉我们它的位置
        label = np.argmax(outputs)
        print(label, "network's answer")
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
    # print(all_values[0])
    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # print(n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))