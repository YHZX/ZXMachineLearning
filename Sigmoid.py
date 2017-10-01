#! /usr/bin/env python
#-*- coding utf-8 -*-
import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def forward(self,x):
        self.x = x
        self.y = self.Sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1-sig)
        return self.dx
