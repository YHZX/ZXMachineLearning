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


class FullConnet
    def __init__(self, l_x,  l_y):
        self.weights = np.random.randn(l_y,l_x)
        self.bias = np.random.randn(1)

    def forward(self,x):
        self.x = x
        self.y = np.dot(self.weights,x)
        return self.y

    def






























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





