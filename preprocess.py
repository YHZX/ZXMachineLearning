#! /usr/bin/env python
#-*- coding utf-8 -*-

import sys
from scipy import misc
import numpy as np

def main():
    l = len(sys.argv)
    if l<2:
        print("eg:python img2pk1.py list.txt dst.py\n convert image to npy")
        return
    src = sys.argv[1]
    dst = sys.argv[2]
    with open(src,'r') as f:
        list = f.readlines()

    data = []
    labels = []
    for i in list:
        name,label = i.strip('\n').split(' ')
        print (name + 'processed')
        img = misc.imread(name)

        img = img/255
        img.resize((img.size),1)

       # print('img:%d', img.shape)

        data.append(img)
        labels.append(int(label))

    print ('write to npy')
    print('datalen:%d, labelslen:%d', len(data), len(labels))
    np.save(dst,[data,labels])
    print('completed')





if __name__ =='__main__':
    main()








