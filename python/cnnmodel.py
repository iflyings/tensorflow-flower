#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: iflyings
import tensorflow as tf


class CnnModel:
    def __init__(self, input_x, n_classes):
        self.name = "cnnmodel"
        self.input_x = input_x
        self.n_classes = n_classes

    def __weight_variable(self, shape, n):
        # tf.truncated_normal(shape, mean, stddev)这个函数产生正态分布，均值和标准差自己设定。
        # shape表示生成张量的维度，mean是均值
        # stddev是标准差,，默认最大为1，最小为-1，均值为0
        initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
        return initial
 
 
    def __bias_variable(self, shape):
        # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return initial
 

    def __conv2d(self, x, W):
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
        # padding 一般只有两个值
        # 卷积层后输出图像大小为：（W+2P-f）/stride+1并向下取整
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def __max_pool_2x2(self, x, name):
        # 池化卷积结果（conv2d）池化层采用kernel大小为3*3，步数也为2，SAME：周围补0，取最大值。数据量缩小了4倍
        # x 是 CNN 第一步卷积的输出量，其shape必须为[batch, height, weight, channels];
        # ksize 是池化窗口的大小， shape为[batch, height, weight, channels]
        # stride 步长，一般是[1，stride， stride，1]
        # 池化层输出图像的大小为(W-f)/stride+1，向上取整
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    # 一个简单的卷积神经网络，卷积+池化层 x2，全连接层x2，最后一个softmax层做分类。
    # 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    def create(self):
        # 第一层卷积
        # 第一二参数值得卷积核尺寸大小，即patch；第三个参数是通道数；第四个是卷积核个数
        with tf.variable_scope('conv1') as scope:
            # 所谓名字的scope，指当绑定了一个名字到一个对象的时候，该名字在程序文本中的可见范围
            w_conv1 = tf.Variable(self.__weight_variable([3, 3, 3, 64], 0.1), name='weights', dtype=tf.float32)
            b_conv1 = tf.Variable(self.__bias_variable([64]), name='biases', dtype=tf.float32)   # 64个偏置值
            # tf.nn.bias_add 是 tf.add 的一个特例:tf.add(tf.matmul(x, w), b) == tf.matmul(x, w) + b
            # h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(images, w_conv1), b_conv1), name=scope.name)
            h_conv1 = tf.nn.relu(self.__conv2d(self.input_x, w_conv1) + b_conv1, name='conv1')  # 得到128*128*64(假设原始图像是128*128)
            pool1 = self.__max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*64
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
        # 第二层卷积
        # 32个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
        with tf.variable_scope('conv2') as scope:
            w_conv2 = tf.Variable(self.__weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
            b_conv2 = tf.Variable(self.__bias_variable([32]), name='biases', dtype=tf.float32)   # 32个偏置值
            h_conv2 = tf.nn.relu(self.__conv2d(norm1, w_conv2) + b_conv2, name='conv2')  # 得到64*64*32
            pool2 = self.__max_pool_2x2(h_conv2, 'pooling2')  # 得到32*32*32
            norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
        # 第三层卷积
        # 16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
        with tf.variable_scope('conv3') as scope:
            w_conv3 = tf.Variable(self.__weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
            b_conv3 = tf.Variable(self.__bias_variable([16]), name='biases', dtype=tf.float32)   # 16个偏置值
            h_conv3 = tf.nn.relu(self.__conv2d(norm2, w_conv3)+b_conv3, name='conv3')  # 得到32*32*16
            pool3 = self.__max_pool_2x2(h_conv3, 'pooling3')  # 得到16*16*16
            norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    
        reshape = tf.reshape(norm3, shape=[self.input_x.shape[0], -1])
        dim = reshape.get_shape()[1].value

        # 第四层全连接层
        # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
        with tf.variable_scope('local3') as scope:
            w_fc1 = tf.Variable(self.__weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
            b_fc1 = tf.Variable(self.__bias_variable([128]), name='biases', dtype=tf.float32)
            h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)
    
        # 第五层全连接层
        # 128个神经元，激活函数relu()
        with tf.variable_scope('local4') as scope:
            w_fc2 = tf.Variable(self.__weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
            b_fc2 = tf.Variable(self.__bias_variable([128]), name='biases', dtype=tf.float32)
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)
    
        # 对卷积结果执行dropout操作
        # keep_prob = tf.placeholder(tf.float32)
        h_fc2_dropout = tf.nn.dropout(h_fc2, rate = 0.5)
        # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
        # 第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符
    
        # Softmax回归层
        # 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是6类，所以这个层输出的是六个得分。
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.Variable(self.__weight_variable([128, self.n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
            biases = tf.Variable(self.__bias_variable([self.n_classes]), name='biases', dtype=tf.float32)
            softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
            # softmax_linear = tf.nn.softmax(tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear'))
        return softmax_linear
