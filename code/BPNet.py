import numpy as np
import pandas as pd
from numpy import dot  # 导入矩阵乘法函数名

'''
Written by Zheng Jiacan, Shenzhen University, 2020/11/1. 

BPNet is a full-connected multilayer network which uses stochastic gradient descent based back 
propagation algorithm for optimization. BPNet can be used for classification or regression tasks. 

This code is a python code of BPNet based on numpy library, where I define some core functions of BPNet. 

Because I mainly untilize the Jacobia matrix for mathematical derivation and python coding, which
is more concise and much easier, it is recommonded that reading the BPNet.pdf firstly before you 
start to read the python code. 
'''

def lossFun(H,Y_hat,fun):
    if(fun == 'crossEntropy'): 
        # 计算 交叉熵 损失函数的值
        # H: 标签矩阵，每一行表示一个样本 one-hot 类别标签行向量， Y_hat: 网络最后输出得到的预测标签矩阵，每一行表示一个样本的概率分布行向量
        # loss = - <H, log Y_hat> /m 
        temp = Y_hat[H == 1] 
        lossValue = -np.mean(np.log(temp)) 
    elif(fun == 'squareLoss'): 
        # 计算 回归任务 中的 最小二乘 损失函数的值 
        # H: 回归矩阵，每一行是一个样本回归向量； Y_hat: 网络最后输出的回归矩阵，每一行是一个预测回归向量 
        # loss = ||H - Y_hat||_F^2 / (2m) 
        temp = np.power(H-Y_hat,2) 
        lossValue = sum(sum(temp))/(2*H.shape[0])
    return lossValue

def activeFun(Z,fun):
    # receive a feature matrix Z, whose each row represents a sample feature, and return a active matrix Y
    if(fun == 'sigmoid'):
        # 计算sigmoid的函数值，矢量化计算, Z可以为一个矩阵
        Z[Z>=0] = 1/(1+np.exp(-Z[Z>=0]))        # 防止溢出处理
        Z[Z<0] = np.exp(Z[Z<0])/(1+np.exp(Z[Z<0]))
    elif(fun == 'tanh'):
        # 计算 tanh 的函数值，矢量化计算, 利用 tanh(z) = 2sigmoid(2z) - 1
        Z = 2*Z
        Z[Z>=0] = 1/(1+np.exp(-Z[Z>=0]))        # 防止溢出处理
        Z[Z<0] = np.exp(Z[Z<0])/(1+np.exp(Z[Z<0]))
        Z= 2*Z - 1
    elif(fun == 'ReLU'):
        Z[Z<0] = 0
    return Z

def softmax(Z): 
    # 计算 softmax 的函数值，矢量化计算, Z可以为一个矩阵
    # 因为 max 操作默认转化为行向量，因此需要 keepdims 来保持其为列向量
    Z = Z - Z.max(axis = 1, keepdims = True)                 # 防止溢出处理
    Z = np.exp(Z)
    Z = Z/Z.sum(axis = 1, keepdims = True)
    return Z
    
def activeFun_G(Y,fun):
    # 本函数计算三个逐元素激活的激活函数的导数，注意到返回的是一个矩阵 G_G（具体推导参考 PDF 文档）
    # Y is a actived matrix. 
    if(fun == 'sigmoid'):
        G_G = Y * (1 - Y)
    elif(fun == 'ReLU'):
        Y[Y>0] = 1
        G_G = Y
    elif(fun == 'tanh'):
        G_G = 1 - np.power(Y,2)
    return G_G

def BPNetTrain(X, H, layerNum, neuronNumList, batchNum, step, actFunction, task):
    '''
    BPNet is a full-connected multilayer network which uses stochastic gradient descent based back propagation 
    algorithm for optimization. 

    The meaning of parameters in the BPNet interface is descripted as following : 

        X : data matrix, each row represents a sample . 
        Y : regression matrix, may be label matrix (one-hot) for classification, or may be the regression matrix. 
        layerNum: The layer number of BPNet. 
        neuronNumList: The corresponding neuron number list. 
        batch: The number of train samples in each stochastic gradient estimation. 
        step: The gradient descent step. 
        actFunction: the choose of active function, which can take three choose : 
                    actFuntion = 'sigmoid', actFuntion = 'tanh', actFuntion = 'ReLU'. 
        task: if classification task, then choose task = 'classification', 
                      if regression task, then choose task = 'regression'. 

    Example : 
        BPNetTrain(X = trainImage, H = labelMatrix, layerNum = 3, neuronNumList = [784, 300 , 10], batchNum = 1000, 
                    step = 0.1, actFunction = 'ReLU', task = 'crossEntropy') 

    注 ： 第一层节点数 784 为输入的原始图像像素点维数，而最后一层的节点数 10 为分类的类别数; 
          因此这是一个三层的网络，只有一个隐藏层，其有 300 个神经元节点。
    '''

    # 参数初始化，注意每一层网络都是设置了 bias 的，因此每一层的矩阵 B 的第一行都是偏置行向量. 
    parametersB = layerNum*[0] # 为了与 pdf 推导的公式一致， 0 索引的位置不用， layerNum = ell + 1
    # 正态分布初始化网络参数  Xavier/He initialization. 
    for i in range(1,layerNum):    # 初始化 [1,...,ell] 
        parametersB[i] = np.random.normal(0,np.sqrt(2/(neuronNumList[i-1]+neuronNumList[i])),(neuronNumList[i-1]+1,neuronNumList[i]))
        parametersB[i][0,:] = 0   # 偏置行初始化为 0. 
    
    alpha = step/batchNum  # 批量梯度需要平均，这部分可以直接一次性体现在步长中
    # 训练网络的迭代参数：总共迭代多少次训练样本
    globalIterMax = 60
    localIterMax = X.shape[0]//batchNum  # 训练样本可以分成多少个 batch. 
    globalIter = 0 
    localIter = 0 
    # 样本索引列表 
    index = list(range(X.shape[0]))
    while(globalIter < globalIterMax):
        print("{0:-^70}".format('  全局迭代次数:'+ str(globalIter)+'/'+str(globalIterMax)+'  '),end = '\n\n')
        # 将所有样本的顺序打乱，即打乱索引列表index 
        np.random.shuffle(index)
        localIter = 0
        while(localIter < localIterMax):
            # 提取本轮批量样本的索引，数量为 batchNum 
            batchIndex = index[localIter*batchNum:(localIter+1)*batchNum] 
            # 数组的花式索引，返回副本 
            # (这里存在优化空间: 直接获得原数据矩阵的内存地址引用, 从而省去了复制副本的时间和空间消耗)
            batchX = X[batchIndex,:]
            batchH = H[batchIndex,:]
            # 前向传播,得到中间结果
            middleZ, middleY = forwardProp(batchX, parametersB, actFunction, task)
            # 计算所有梯度的平均 
            B_G = backProp(batchX, batchH, middleZ, middleY, parametersB, actFunction, task) 
            # 更新参数, 梯度下降公式 
            for i in range(1,layerNum): 
                parametersB[i] = parametersB[i] - alpha*B_G[i].T
            # 更新样本内迭代次数
            localIter = localIter + 1
        # 更新样本外迭代次数
        globalIter = globalIter + 1
        # 一轮训练样本迭代结束后，查看当前参数对任务的适应情况
        Y_hat = BPNetMap(X, parametersB, actFunction, task)
        showResult(H,Y_hat,task)
    return parametersB

def backProp(X, H, middleZ, middleY, parametersB, actFunction, task):
    # 计算后向传播的梯度 
    layerNum = len(parametersB)   # layerNum = ell + 1
    B_G = layerNum*[0] 

    # 计算顶层梯度 Z_G, 这个顶层梯度是后向传播算法的初始输入（关键），其由损失函数等决定
    if(task == 'classification'):
        Z_G = middleY[layerNum - 1] - H
    elif(task == 'regression'): 
        Z_G = middleY[layerNum - 1] - H

    # 计算出顶层的梯度 Z_G 之后，按照 PDF 的三个后向传播公式计算
    for i in range(layerNum-1,1,-1):
        B_G[i] = dot( Z_G.T, middleY[i-1] )
        Y_G = dot( Z_G, parametersB[i][1:,:].T )    # 注意要把 Y_G 的第一列去除，等价于去除 parametersB[i] 的第一行（偏置行），详见 PDF 推导
        G_G = activeFun_G( middleY[i-1][:,1:], actFunction )  # 注意把 middleY[i-1] 的第一列去除
        Z_G = Y_G * G_G 

    # 最后一层导数
    B_G[1] = dot( Z_G.T,  middleY[0])
    return B_G
    
def forwardProp(X, parametersB, actFunction, task):
    # 计算并返回前向传播的输出值和一些中间结果的值
    # 矢量化计算，X可以是样本矩阵，其中X的每一行表示一个样本
    layerNum = len(parametersB)   # 网络层数 
    middleY = layerNum*[0] 
    middleY[0] = np.hstack( (np.ones((X.shape[0],1)),X) )
    middleZ = layerNum*[0] 

    # 网络层前向传播 [1,...,ell-1]
    for i in range(1,layerNum-1):
        middleZ[i] = dot(middleY[i-1], parametersB[i])
        middleY[i] = activeFun(middleZ[i], fun = actFunction)
        middleY[i] = np.hstack( (np.ones((middleY[i].shape[0],1)), middleY[i]) )
    
    # 最后一层网络
    middleZ[layerNum-1] = dot(middleY[layerNum-2], parametersB[layerNum-1])  # 线性映射
    if(task == 'classification'): 
        # softmax 映射
        middleY[layerNum-1] = softmax(middleZ[layerNum-1])
    elif(task == 'regression'):
        # indentity 映射
        middleY[layerNum-1] = middleZ[layerNum-1]
    
    return middleZ,middleY

def BPNetMap(X, parametersB, actFunction, task): 
    # 学习到的网络映射
    layerNum = len(parametersB)   # 网络层数 
    Y = np.hstack( (np.ones((X.shape[0],1)),X) )

    # 网络层前向传播 [1,...,ell-1]
    for i in range(1,layerNum-1):
        Z = dot(Y, parametersB[i])
        Y = activeFun(Z, fun = actFunction)
        Y = np.hstack( (np.ones((Y.shape[0],1)),Y) ) 
    
    # 最后一层网络 
    Z = dot(Y, parametersB[layerNum-1])  # 线性映射
    if(task == 'classification'): 
        # softmax 映射
        Y_hat = softmax(Z)
    elif(task == 'regression'):
        # indentity 映射
        Y_hat = Z 
    
    return Y_hat

    
def showResult(H,Y_hat,task):
    # 矢量化计算，X可以是样本矩阵，每一列表示一个样本
    
    if(task == 'classification'):
        # 分析正确率和分类概率的平均值
        predictLabel = Y_hat.argmax(axis = 1)
        result_bool = H[np.arange(H.shape[0]),predictLabel]
        correctRate = sum(result_bool)/H.shape[0]
        lossFunValue = lossFun(H,Y_hat,'crossEntropy')
        print('{0:<26}{1:<0}'.format('分类正确率：'+str(correctRate),'损失函数值：'+str(lossFunValue)),end = '\n\n')
    elif(task == 'regression'):
        lossFunValue = lossFun(H,Y_hat,'squareLoss')
        print('{0:<26}'.format('回归损失函数值：'+str(lossFunValue)), end = '\n\n')
    return 0