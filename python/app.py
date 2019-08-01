#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: iflyings
import os
import tensorflow as tf
from recordres import RecordRes
from imageres import ImageRes
from cnnmodel import CnnModel
from cnnnetwork import CnnNetwork

# 传入参数：logits，网络计算输出值。labels，真实值，0或者1
# 返回参数：loss，损失值
def loss(logits, label_batches):
    with tf.variable_scope('loss') as scope:
        # 使用交叉熵损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_batches, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss

# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        return train_op


# 评价/准确率计算
# 输入参数：logits，网络计算值。label_batches，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, label_batches):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, label_batches, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy


def test(image_arr):
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 28, 28, 3])
        # print(image.shape)
        p = deep_CNN(image, 1, N_CLASSES)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[28, 28, 3])
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
            print('Loading success')
        prediction = sess.run(logits, feed_dict={x: image_arr})
        max_index = np.argmax(prediction)
        print('预测的标签为：', max_index, lists[max_index])
        print('预测的结果为：', prediction)


def train():
    image_width = 100
    image_heigh = 100
    batch_size = 50
    learning_rate = 2e-5
    num_epochs = 20000
    image_path = '/home/iflyings/VSCodeProjects/flower/data'
    model_path = '/home/iflyings/VSCodeProjects/flower/model'

    imageRes = ImageRes(image_path, batch_size = batch_size, image_width = image_width, image_height = image_heigh)
    image_batch = imageRes.load()
    # 训练
    #cnnModel = CnnModel(image_batch[0], imageRes.n_classes)
    cnnModel = CnnNetwork(image_batch[0], imageRes.n_classes)
    train_logits = cnnModel.create()
    train_loss = loss(train_logits, image_batch[1])
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, image_batch[1])
    # 这个是log汇总记录
    summary_op = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    train_writer = tf.compat.v1.summary.FileWriter(model_path, sess.graph)
    saver = tf.compat.v1.train.Saver() # 产生一个saver来存储训练好的模型

    coord = tf.train.Coordinator()# 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)# 启动QueueRunner, 此时文件名队列已经进队
    try:
        # 执行MAX_STEP步的训练，一步一个batch
        for epoch in range(num_epochs):
            if coord.should_stop():
                break
            _, tra_acc, tra_loss = sess.run([train_op, train_acc, train_loss])
        
            if epoch % 100 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (epoch, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, epoch)
            # 保存训练好的模型
            if epoch % (num_epochs / 10) == (num_epochs / 10) - 1:
                checkpoint_path = os.path.join(model_path, 'thing.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()# When done, ask the threads to stop.
    coord.join(threads)

    sess.close()


if __name__ == '__main__':
    train()
