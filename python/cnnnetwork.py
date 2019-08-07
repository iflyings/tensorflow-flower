#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import tensorflow as tf

class CnnNetwork:
    def __init__(self, input_x, batch_size, n_classes, train):
        self.name = "cnnnetwork"
        self.input_x = input_x
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.train = train

    def __conv_wrapper(self, input, filters=32, train=True, activation=tf.nn.relu):
        conv = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=[3, 3],
            padding="same",
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            name="conv")
        bn = tf.layers.batch_normalization(conv,
            momentum=0.9,
            epsilon=1e-5,
            scale=True,
            training=train,
            name="bn")
        return tf.nn.relu(bn)

    def __pool_wrapper(self, input, name='pool'):
        return tf.layers.max_pooling2d(
                inputs=input, 
                pool_size=[2, 2], 
                strides=2,
                name="pool")

    def __dense_wrapper(self, input, units):
        return tf.layers.dense(inputs=input,
                units=units,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                name="dense")

    def create(self):
        # 第一个卷积层
        with tf.compat.v1.variable_scope('conv1') as scope:
            conv1 = self.__conv_wrapper(self.input_x,filters=32,train=True)
            pool1 = self.__pool_wrapper(conv1)
        # 第二个卷积层
        with tf.compat.v1.variable_scope('conv2') as scope:
            conv2 = self.__conv_wrapper(pool1,filters=64,train=True)
            pool2 = self.__pool_wrapper(conv2)
        # 第三个卷积层
        with tf.compat.v1.variable_scope('conv3') as scope:
            conv3 = self.__conv_wrapper(pool2,filters=128,train=True)
            pool3 = self.__pool_wrapper(conv3)
        # 第四个卷积层
        with tf.compat.v1.variable_scope('conv4') as scope:
            conv4 = self.__conv_wrapper(pool3,filters=128,train=True)
            pool4 = self.__pool_wrapper(conv4)

        #dense0 = tf.reshape(pool4, [-1, 6 * 6 * 128])
        flatten = tf.contrib.layers.flatten(pool4)
        # 防止过拟合，加入dropout
        #re1 = tf.layers.dropout(inputs=re1, rate=0.5)

        # 全连接层
        with tf.compat.v1.variable_scope('dense1') as scope:
            dense1 = self.__dense_wrapper(flatten, 512)
        with tf.compat.v1.variable_scope('dense2') as scope:
            dense2 = self.__dense_wrapper(dense1, 256)
        with tf.compat.v1.variable_scope('dense3') as scope:
            logits = self.__dense_wrapper(dense2, self.n_classes)
            
        return logits
        ### 四个卷积层，两个全连接层，一个softmax层组成。
        ### 在每一层的卷积后面加入 batch_normalization, relu, 池化
        ### batch_normalization 层很好用，加了它之后，有效防止了梯度消失和爆炸，还加速了收敛。