#! /usr/bin/env python
#-*- coding utf-8 -*-
import numpy as np

class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.accuracy = np.sum([np.argmax(xx) == ll  for xx, ll in zip(x,label)])
        self.accuracy = 1.0 *self.accuracy /x.shape[0]
        return self.accuracy