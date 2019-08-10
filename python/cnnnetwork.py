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

    def __conv_wrapper(self, input, filters=32, train=True, activation=tf.nn.relu, name="1"):
        conv = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=[3, 3],
            padding="same",
            activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            name="conv_"+name)
        '''
        bn = tf.layers.batch_normalization(conv,
            momentum=0.9,
            epsilon=1e-5,
            scale=True,
            training=train,
            name="bn_"+name)
        '''
        return conv

    def __pool_wrapper(self, input, name='1'):
        return tf.layers.max_pooling2d(
                inputs=input, 
                pool_size=[2, 2], 
                strides=2,
                name="pool_"+name)

    def __dense_wrapper(self, input, units, activation=tf.nn.relu, name='1'):
        return tf.layers.dense(inputs=input,
                units=units,
                activation=activation,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                name="dense_"+name)

    def create(self):
        # 第一个卷积层
        #with tf.compat.v1.variable_scope('conv1') as scope:
        conv1_1 = self.__conv_wrapper(self.input_x,filters=32,train=True,name="conv1_1")
        pool1 = self.__pool_wrapper(conv1_1,name="pool1")
        # 第二个卷积层
        #with tf.compat.v1.variable_scope('conv2') as scope:
        conv2_1 = self.__conv_wrapper(pool1,filters=64,train=True,name="conv2_1")
        pool2 = self.__pool_wrapper(conv2_1,name="pool2")
        # 第三个卷积层
        #with tf.compat.v1.variable_scope('conv3') as scope:
        conv3_1 = self.__conv_wrapper(pool2,filters=128,train=True,name="conv3_1")
        pool3 = self.__pool_wrapper(conv3_1,name="pool3")
        # 第四个卷积层
        #with tf.compat.v1.variable_scope('conv4') as scope:
        conv4_1 = self.__conv_wrapper(pool3,filters=256,train=True,name="conv4_1")
        pool4 = self.__pool_wrapper(conv4_1,name="pool4")
        #dense0 = tf.reshape(pool4, [-1, 6 * 6 * 128])
        flatten = tf.layers.flatten(pool4)
        # 防止过拟合，加入dropout
        #re1 = tf.layers.dropout(inputs=re1, rate=0.5)

        # 全连接层
        #with tf.compat.v1.variable_scope('dense1') as scope:
        dense1 = self.__dense_wrapper(flatten,512,name="dense1")
        #with tf.compat.v1.variable_scope('dense2') as scope:
        dense2 = self.__dense_wrapper(dense1,256,name="dense2")
        #with tf.compat.v1.variable_scope('dense3') as scope:
        logits = self.__dense_wrapper(dense2,self.n_classes,name="logits")

        return logits
        ### 四个卷积层，两个全连接层，一个softmax层组成。
        ### 在每一层的卷积后面加入 batch_normalization, relu, 池化
        ### batch_normalization 层很好用，加了它之后，有效防止了梯度消失和爆炸，还加速了收敛。