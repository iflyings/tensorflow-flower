#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import os
import cv2
import numpy as np
import tensorflow as tf

# 文件夹名
str_2 = './train_cifar10'
str_1 = './test_cifar10'

if os.path.exists(str_1) == False:
    os.mkdir(str_1)
if os.path.exists(str_2) == False:
    os.mkdir(str_2)


def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def cifar_jpg(dir_file, file_list):
    for file in file_list:
        dataName = dir_file + '/' + file
        Xtr = unpickle(dataName)
        print(len(Xtr[b'labels']))
        print(dataName + " is loading...")
        for i in range(0, 10000):
            img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            label = str(Xtr[b'labels'][i])
            if not os.path.exists('./train_cifar10/' + label):
                os.mkdir('./train_cifar10/' + label)
            filename = str(Xtr[b'filenames'][i])
            filename = filename.split('.')[0] + '.jpg'
            picpath = './train_cifar10/' + label + '/' + filename
            cv2.imwrite(picpath, img)
        print(dataName + " loaded.")

if __name__ == '__main__':
    dir_file = './cifar-10-batches-py'
    file_list = ('data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch')
    cifar_jpg(dir_file, file_list)