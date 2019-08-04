#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import tensorflow as tf

class CnnNetwork:
    def __init__(self, input_x, n_classes):
        self.name = "cnnnetwork"
        self.input_x = input_x
        self.n_classes = n_classes

    def __batch_norm(self, x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
        return tf.layers.batch_normalization(x,
                        momentum=momentum,
                        epsilon=epsilon,
                        scale=True,
                        training=train,
                        name=name)

    def create(self):
        # 第一个卷积层（100——>50)
        conv1 = tf.layers.conv2d(
            inputs=self.input_x,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #conv1 = self.__batch_norm(conv1, name='pw_bn1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # 第二个卷积层(50->25)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #conv2 = self.__batch_norm(conv2, name='pw_bn2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # 第三个卷积层(25->12)
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #conv3 = self.__batch_norm(conv3, name='pw_bn3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # 第四个卷积层(12->6)
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        #conv4 = self.__batch_norm(conv4, name='pw_bn4')
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

        re1 = tf.reshape(pool4, [self.input_x.shape[0], -1])

        # 防止过拟合，加入dropout
        dropout = tf.layers.dropout(inputs=re1, rate=0.5)

        # 全连接层
        dense1 = tf.layers.dense(inputs=dropout,
                                units=1024,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        dense2 = tf.layers.dense(inputs=dense1,
                                units=512,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        logits = tf.layers.dense(inputs=dense2,
                                units=self.n_classes,
                                activation=None,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        return logits
        ### 四个卷积层，两个全连接层，一个softmax层组成。
        ### 在每一层的卷积后面加入batch_normalization, relu, 池化
        ### batch_normalization层很好用，加了它之后，有效防止了梯度消失和爆炸，还加速了收敛。