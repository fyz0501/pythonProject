import numpy as np
from scipy.spatial import distance


def data_handle(data, classTag):
    x_add = [[1] for i in range(0, len(data))]
    data = np.array([data+x_add for data, x_add in zip(data, x_add)])
    for index in range(0, len(data)-1):
        if classTag[index] == 0:
            data[index] = np.negative(data[index])
            print("y[{}]={}".format(index, data[index]))
    print("y={}".format(data))
    w_init = np.zeros(len(data[0]))
    w_init[0] = 1
    return w_init, data


# single sample
def perceptron(a, y, classTag):
    end_flag = 1
    print("")
    while end_flag:
        end_flag = 0
        for index in range(0, len(y)):
            print("使用的样本为：{}".format(y[index]))
            if (np.dot(a, y[index])) < 0:
                end_flag = 1
                a += y[index]
                print("分类结果为：{},分类错误".format(0 and classTag[index]))
                print("w = {}, w_0 = {}".format(a[:-1], a[-1]))
            else:
                print("分类结果为：{},分类正确".format(classTag[index]))
                print("w = {}, w_0 = {}".format(a[:-1], a[-1]))


# 双曲正切sigmoid函数
def sigmoid(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def d_sigmoid(x):
    return 1-x**2


def forward(x, weights, biases):
    for w, b in zip(weights, biases):
        # x[2,4] 输入之前需要转置
        # return[2,4]
        z = np.dot(w, x) + b
        x = sigmoid(z)  # 将隐含层的sigmoid输出作为输出层的输入再进行循环
    return x


# t:真实数据 x==>[2:1]//一次输入一个样本
def forward_backward(x, t, weights, biases, sizes):
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b = [np.zeros(b.shape) for b in biases]
    # 先进行一次forward，保存相关信息用于backward
    activations = [x]   # activations用来保存每一次激活函数作用后的值
    zs = []
    activation = x
    for w, b in zip(weights, biases):
        z = np.dot(w, activation) + b
        activation = sigmoid(z)
        zs.append(z)
        activations.append(activation)
    # backward
    # 1. 计算输出层的梯度
    delta_k = d_sigmoid(activations[-1])*(activations[-1] - t)  # 计算出输出层的\delta_k
    nabla_b[-1] = delta_k
    nabla_w[-1] = np.dot(delta_k, activations[-2].T)  # 输出层中w的梯度为\delta_k * 上一层sigmoid输出的结果

    # 2. 计算隐藏层的梯度
    # 用for循环是为了封装成不仅是一层隐藏层的情况
    for layer in range(2, len(sizes)):
        layer = -layer
        z = zs[layer]
        a = activations[layer]
        delta_j = np.dot(weights[layer+1].T, delta_k) * d_sigmoid(a) * -1

        nabla_b[layer] = delta_j
        nabla_w[layer] = np.dot(delta_j, activations[layer-1].T)

    return nabla_w, nabla_b


# 误差函数
def cal_E(res, t):
    e = 0
    for i in range(0, len(res)):
        e += 0.5*(distance.euclidean(res[i], t[i])**2)
    return e


def mul_perceptron(training_data, t, sizes, lr, end_theta):
    # sizes ==> [2, 3, 2]
    # training_data ==> [4,2]
    # t ==> [4,2]
    # w:[ch_out, ch_in] ==》 [3,2],[2,3]
    weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])] # [2,3], [3,2]
    biases = [np.random.randn(ch) for ch in sizes[1:]]
    res_init = forward(training_data.T, weights, biases)
    e = cal_E(res_init, t)  # 初始化误差
    while e > end_theta:
        e = 0
        nabla_w = 0
        nabla_b = 0
        for x, ans in training_data, t:
            nabla_w_, nabla_b_ = forward_backward(x, ans, weights, biases, sizes)
            nabla_w += nabla_w_
            nabla_b += nabla_b_
        nabla_w = nabla_w/len(training_data)
        nabla_b = nabla_b/len(t)
        weights = [w - lr * nabla for w, nabla in zip(weights, nabla_w)]
        biases = [b - lr * nabla for b, nabla in zip(biases, nabla_b)]
        res = forward(training_data, weights, biases)
        e = cal_E(res, t)
    return weights, biases
