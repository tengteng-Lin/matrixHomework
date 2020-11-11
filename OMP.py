#coding:utf-8
import math,time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

def cs_omp(y,A,K):
    '''

    :param y: 观测数据向量
    :param A: 字典矩阵
    :param K: 稀疏性指标
    :return: 稀疏的信号向量x
    '''

    N = A.shape[1]
    residual = y  #初始化残差
    index = np.zeros(N, dtype=int)
    for i in range(N):
        index[i] = -1
    result = np.zeros((N, 1))
    for j in range(K):
        pos = np.argmax(np.fabs(np.dot(A.T,residual)))
        index[pos] = 1
        ASelected = A[:, index>=0]

        xk = np.dot(np.linalg.pinv(ASelected), y)
        residual = y-np.dot(ASelected, xk) #更新残差

    result[index >= 0] = xk
    return result


def experiment_1():
    # experiment 1
    k = 8
    m = 20
    n = 50
    x = np.random.randn(n, 1)
    x[:n - k] = 0
    np.random.shuffle(x)

    A = np.random.randn(m, n)
    y = np.dot(A, x)

    start = time.perf_counter()
    result = cs_omp(y, A, k)
    end = time.perf_counter()

    print(result)
    resultx = range(n)

    plt.figure(1)
    plt.title("Input:x")
    for a, b in zip(resultx, x):
        if (b != 0):
            plt.text(a, b, b, ha='center', va='bottom')
    plt.plot(resultx, x)
    plt.show()

    plt.figure(2)
    plt.plot(resultx, result)
    plt.title("y=Ax的K-稀疏解")
    for a, b in zip(resultx, result):
        if (b != 0):
            plt.text(a, b, b, ha='center', va='bottom')
    plt.show()

    print("Runninh time: %s seconds" % (end - start))

    # 验证阶段
    yy = np.dot(A, result)
    plt.figure(3)
    plt.plot(range(m), y, label='观测信号')
    plt.plot(range(m), yy, label='还原信号')
    plt.legend()
    plt.title("信号还原对比")
    plt.show()

def experiment_2():
    # experiment 2
    k = 8
    m = 20
    n = 50
    x = np.random.randn(n, 1)
    x[:n - k] = 0
    np.random.shuffle(x)

    A = np.random.randn(m, n)
    y = np.dot(A, x)

    start = time.perf_counter()
    cs_omp(y, A, k)
    end = time.perf_counter()

    print("Runninh time: %s seconds" % (end - start))



experiment_1()
# experiment_2()