#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import tensorflow as tf

class VGG16:
    def __init__(self, n_classes, is_train = True):
        self.is_train = is_train
        self.n_classes = n_classes

    def __conv_wrapper(self, inputs, filters, activation=tf.nn.relu, name="conv"):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[3, 3],
            padding="same",
            activation=activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            name=name)
        '''
        bn = tf.layers.batch_normalization(conv,
            momentum=0.9,
            epsilon=1e-5,
            scale=True,
            training=self.is_train,
            name="bn_"+name)
        '''
        return conv

    def __pool_wrapper(self, inputs, name="pool"):
        return tf.layers.max_pooling2d(
                inputs=inputs, 
                pool_size=[2, 2], 
                strides=2,
                name=name)

    def __dense_wrapper(self, inputs, units, activation=tf.nn.relu, name="dense"):
        return tf.layers.dense(
                inputs=inputs,
                units=units,
                activation=activation,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                name=name)

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
            dropout1 = tf.layers.dropout(fc1,training=self.is_train,name="dropout1")
        with tf.compat.v1.variable_scope('layer_7') as scope:
            fc2 = self.__dense_wrapper(dropout1, 4096, name="fc2")
            dropout2 = tf.layers.dropout(fc2,training=self.is_train,name="dropout2")
        with tf.compat.v1.variable_scope('layer_8') as scope:
            fc3 = self.__dense_wrapper(dropout2, self.n_classes, activation=None, name="fc3")
        return fc3