import os 
import sys

import numpy as np
import pandas as pd
from BPNet import *

'''
    This code uses minist dataset to test BPNet. 
'''


def main():
    # 读取 minist 手写数字数据集 
    # path = os.path.abspath(os.path.dirname(sys.argv[0]))      # 获取当前路径名
    path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),os.path.pardir))      # 获取当前路径的父级目录路径名 （上一级目录）    
    trainImage = pd.read_csv(path + '/data/minist_images_train.csv',
                       header = None)
    trainLabel = pd.read_csv(path + '/data/minist_labels_train.csv',
                       header = None)
    #将dataframe类型转化为矩阵类型并转置，方便操作
    trainImage = np.array(trainImage)#[:1000,:]               # 60000*784  S = {0,1,...,254,255} 一个字节表示一个像素，共 60000 个样本，每个样本 784 = 28*28 维度
    trainLabel = np.array(trainLabel)#[:1000,:]               # 60000*1    S = {0,1,...,8,9}

    # 转化成 one-hot 标签矩阵 H 
    trainLabelMatrix = np.zeros((trainLabel.shape[0], 10)) 
    trainLabelMatrix[np.arange(trainLabel.shape[0]),trainLabel[:,0]] = 1  # 此处 trainLabel[:,0] 是为了转化成以为数组，不然有莫名的 bug... 
    
    #测试数据读取
    testImage = pd.read_csv(path + '/data/minist_images_test.csv',
                       header = None)
    testLabel = pd.read_csv(path + '/data/minist_labels_test.csv',
                       header = None)
    #将dataframe类型转化为矩阵类型并转置，方便操作
    testImage = np.array(testImage)               # 10000*784  S = {0,1,...,254,255} 一个字节表示一个像素
    testLabel = np.array(testLabel)               # 10000*1    S = {0,1,...,8,9}
    
    # 转化成 one-hot 标签矩阵 H 
    testLabelMatrix = np.zeros((testLabel.shape[0], 10)) 
    testLabelMatrix[np.arange(testLabel.shape[0]),testLabel[:,0]] = 1    
    
    # 训练 BPNet
    X = trainImage
    H = trainLabelMatrix
    layerNum = 3
    # neuronNumList = [784, 300, 10] for classification
    # neuronNumList = [784, 10, 784] for AutoEncoding. 
    neuronNumList = [784, 300, 10]  

    batchNum = 500 
    # for classification, step = 0.1 for sigmoid or tanh, step = 0.0001 for ReLU
    # for regression, step = 0.01 for sigmoid or tanh, step = 0.0001 for ReLU 
    # noted: The reason that why ReLU should have a smaller step is that ReLU always have a much greater gradient. 
    step = 0.1
    actFunction = 'sigmoid'   

    task = 'classification'
    parametersB = BPNetTrain(X, H, layerNum, neuronNumList, batchNum, step, actFunction, task)

    # task = 'regression'
    # parametersB = BPNetTrain(X, X, layerNum, neuronNumList, batchNum, step, actFunction, task)
    
    # 测试数据上的结果
    Y_hat = BPNetMap(testImage, parametersB, actFunction, task)
    # show test result
    print('test data result: ', end = '\n\n') 
    showResult(testLabelMatrix, Y_hat, task) 
    
    # regression test result
    # showResult(testImage, Y_hat, task) 

    # 保存参数的训练结果
    np.save(path + '/trainParameters/'+ str(len(parametersB)) + '_layers_' + str(neuronNumList) + '_batchNum_' + str(batchNum) 
            + '_step_' + str(step) + '_' + actFunction + '_' + task + '_Net_Parameters', parametersB)

if __name__ == "__main__":
    main()