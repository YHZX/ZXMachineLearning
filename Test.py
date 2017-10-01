#! /usr/bin/env python
#-*- coding utf-8 -*-
from pylab import *

# x = np.linspace(-np.pi, np.pi, 256,  endpoint=True)
# C, S = np.cos(x), np.sin(x)
#
# plot(x,C)
# plot(x,S)
# show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-20,20,1000)
y_t = sigmoid(x)
y = y_t*(1-y_t)

plot(x,y_t)
show()