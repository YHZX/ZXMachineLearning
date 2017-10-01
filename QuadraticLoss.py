#! /usr/bin/env python
#-*- coding utf-8 -*-
import numpy as np

class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for a, b in zip(self.label, label):
            a[b] = 1.0
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0]/2
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]
        return self.dx







