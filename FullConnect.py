#! /usr/bin/env python
#-*- coding utf-8 -*-
import numpy as np
class FullyConnect:
    def __init__(self, l_x, l_y):
        self.weights = np.random.randn(l_y, l_x)/np.sqrt(l_x)
        self.bias = np.random.randn(l_y,1)
        self.lr = 0

    def forward(self,x):
        self.x = x
        self.y = np.array([np.dot(self.weight, xx) + self.bias for xx in x])
        return self.y

    def backward(self,d):
        ddw = [np.dot(dd, xx.T) for dd,xx in zip(d, self.x)]
        self.dw = np.sum(ddw, axis=0)/self.x.shape[0]
        self.db = np.sum(d,axis=0)/self.x.shape[0]
        self.dx = np.array([np.dot(self.weight.T, dd)  for dd in d])

        self.weight -= self.lr*self.dw
        self.bias -= self.lr * self.db
        return self.dx







