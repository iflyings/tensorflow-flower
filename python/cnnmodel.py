#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import tensorflow as tf


class CnnModel:
    def __init__(self, input_x, n_classes):
        self.name = "cnnmodel"
        self.input_x = input_x
        self.n_classes = n_classes
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


    def __conv_wrapper(self, inputs, filters, name):
        filters_val = tf.Variable(tf.truncated_normal(shape=[3,3,inputs.shape[3].value,filters]), name=name+'_filters', dtype=tf.float32)
        biases_val = tf.Variable(tf.constant(0.1, shape=[filters]), name=name+'_biases', dtype=tf.float32)
        conv = tf.nn.conv2d(inputs, filter=filters_val, strides=[1, 1, 1, 1], padding='SAME', name=name+'_conv')
        return tf.nn.relu(conv + biases_val, name=name+'relu')  # 得到128*128*64(假设原始图像是128*128)

    def __pool_wrapper(self, inputs, name='pool'):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __dense_wrapper(self, inputs, units, activation=tf.nn.relu, name="dense"):
        weights_val = tf.Variable(tf.truncated_normal(shape=[inputs.shape[1].value, units]),name=name+'weights', dtype=tf.float32)
        biases_val = tf.Variable(tf.constant(0.1, shape=[units]), name=name+'biases', dtype=tf.float32)
        outputs_val = tf.matmul(inputs, weights_val) + biases_val
        if activation:
            outputs_val = activation(outputs_val, name=name+'relu')
        return outputs_val

    # 一个简单的卷积神经网络，卷积+池化层 x2，全连接层x2，最后一个softmax层做分类。
    # 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    def create(self, inputs):
        with tf.compat.v1.variable_scope('layer_1') as scope:
            conv1_1 = self.__conv_wrapper(inputs, filters=64, name="conv1_1")
            conv1_2 = self.__conv_wrapper(conv1_1, filters=64, name="conv1_2")
            pool1 = self.__pool_wrapper(conv1_2, name="pool1")
        with tf.compat.v1.variable_scope('layer_2') as scope:
            conv2_1 = self.__conv_wrapper(pool1, filters=128, name="conv2_1")
            conv2_2 = self.__conv_wrapper(conv2_1, filters=128, name="conv2_2")
            pool2 = self.__pool_wrapper(conv2_2, name="pool2")
        with tf.compat.v1.variable_scope('layer_3') as scope:
            conv3_1 = self.__conv_wrapper(pool2, filters=256, name="conv3_1")
            conv3_2 = self.__conv_wrapper(conv3_1, filters=256, name="conv3_2")
            conv3_3 = self.__conv_wrapper(conv3_2, filters=256, name="conv3_3")
            pool3 = self.__pool_wrapper(conv3_3, name="pool3")
        with tf.compat.v1.variable_scope('layer_4') as scope:
            conv4_1 = self.__conv_wrapper(pool3, filters=512, name="conv4_1")
            conv4_2 = self.__conv_wrapper(conv4_1, filters=512, name="conv4_2")
            conv4_3 = self.__conv_wrapper(conv4_2, filters=512, name="conv4_3")
            pool4 = self.__pool_wrapper(conv4_3, name="pool4")
        with tf.compat.v1.variable_scope('layer_5') as scope:
            conv5_1 = self.__conv_wrapper(pool4, filters=512, name="conv5_1")
            conv5_2 = self.__conv_wrapper(conv5_1, filters=512, name="conv5_2")
            conv5_3 = self.__conv_wrapper(conv5_2, filters=512, name="conv5_3")
            pool5 = self.__pool_wrapper(conv5_3, name="pool5")

        flatten = tf.layers.flatten(pool5)

        with tf.compat.v1.variable_scope('layer_6') as scope:
            fc1 = self.__dense_wrapper(flatten, 4096, name="fc1")
        with tf.compat.v1.variable_scope('layer_7') as scope:
            fc2 = self.__dense_wrapper(fc1, 4096, name="fc2")
        with tf.compat.v1.variable_scope('layer_8') as scope:
            fc3 = self.__dense_wrapper(fc2, self.n_classes, activation=None, name="fc3")
        return fc3
