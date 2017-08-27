#coding=utf-8
#__author__='change'
import numpy as np
# print(np.__version__)
#
# array=np.array([1,2,3], dtype=np.complex64)
# print(array)
#
# mat1=np.zeros((2,3),dtype=complex)
# print(mat1)
#
# nd=np.zeros((2,3,4,5))
# print(nd.shape)
# print(nd.size)
#
#
# mat2=np.ones((2,3))
# scalar = 4
# mat3= mat2*scalar
# print(mat3)
#
# mat4=mat2.T
# print(mat4.shape, mat4.size)
#
# print('n dimentional array')
# mat5=np.ones((2,3,5))
# mat6=mat5.T
# print(mat6.shape, mat6.size)
# print(mat6)
#
# mat7=np.ones((2,3))
# mat8=np.ones((3,5))
# mat9=mat7.dot(mat8)
# print(mat9)
#
#
#
# print('broadcast')
# mat10=np.zeros((3,2))
# vec=np.array([[1],[2],[3]])
# print(mat10+vec)


# rannum=np.random.randn(2,3)
# print(rannum)
#
#
# print('axis')
# a=np.ones((3,2))
# print(np.sum(a,axis=0))
# print(np.sum(a,axis=1))
#
# print('exp')
# mata1=np.array([1,0])
# print(np.exp(mata1))


class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x)  # 使用随机数初始化参数
        self.bias = np.random.randn(1)  # 使用随机数初始化参数

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.dot(self.weights, x) + self.bias  # 计算w11*a1+w12*a2+bias1
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        self.dw = d * self.x  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.db = d
        self.dx = d * self.weights
        return self.dw, self.db  # 返回求得的参数梯度，注意这里如果要继续反向传递梯度，应该返回self.dx


class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self):  # 这里sigmoid是最后一层，所以从这里开始反向计算梯度
        sig = self.sigmoid(self.x)
        self.dx = sig * (1 - sig)
        return self.dx  # 反向传递梯度


def main():
    fc = FullyConnect(2, 1)
    sigmoid = Sigmoid()
    x = np.array([[1], [2]])
    print ('parameters: weights:', fc.weights, ' bias:', fc.bias, ' input: ', x)

    # 执行前向计算
    y1 = fc.forward(x)
    y2 = sigmoid.forward(y1)
    print ('forward result: ', y2)

    # 执行反向传播
    d1 = sigmoid.backward()
    dx = fc.backward(d1)

    print('sigmoid back result: ', d1)
    print ('backward result: ', dx)


if __name__ == '__main__':
    main()






























#
# import numpy as np
# a= np.array([1,2,3], dtype=complex)
# print (a)
#
#
# from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# # 计算正弦曲线上点的 x 和 y 坐标
# x = np.arange(0,  3  * np.pi,  0.1)
# y = np.sin(x)
# plt.title("sine wave form")
# # 使用 matplotlib 来绘制点
# plt.plot(x, y,'^r')
# plt.show()





