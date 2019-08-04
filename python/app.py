#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import os
import numpy as np
import tensorflow as tf
from recordres import RecordRes
from imageres import ImageRes
from cnnmodel import CnnModel
from cnnnetwork import CnnNetwork

image_width = 100
image_heigh = 100
batch_size = 20
learning_rate = 0.001
step_count = 5000
image_path = './data'
model_path = './model'

# 传入参数：logits，网络计算输出值。labels，真实值，0或者1
# 返回参数：loss，损失值
def loss(logits, label_batches):
    with tf.compat.v1.variable_scope('loss') as scope:
        # 使用交叉熵损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_batches, name='xentropy_per_example')
        return tf.reduce_mean(cross_entropy, name='loss')

# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.compat.v1.variable_scope('optimizer'):
        return tf.compat.v1.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)
        #return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 评价/准确率计算
# 输入参数：logits，网络计算值。label_batches，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, label_batches):
    with tf.compat.v1.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, label_batches, 1)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

def test(test_dataset, n_classes):
    test_iterator = test_dataset.make_initializable_iterator()

    train_x = tf.placeholder(shape=[1,image_width,image_heigh,3],dtype=tf.float32)
    cnnModel = CnnNetwork(train_x, n_classes)
    test_logits = cnnModel.create()
    test_logits = tf.reshape(test_logits, [n_classes])
    test_predict = tf.nn.softmax(test_logits)

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(test_iterator.make_initializer(test_dataset))
    #with tf.Graph().as_default():
    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('can not find train model data')
        return

    print('loading train model data success')
    acc_count = 0
    try:
        for step in range(1000):
            image, label = sess.run(test_iterator.get_next())
            pred_y = sess.run([test_predict], feed_dict={ train_x:image })
            pred_y = np.reshape(pred_y[0], -1)
            pred_y = np.argmax(pred_y)
            if pred_y == label:
                acc_count = acc_count + 1
            print('%03d 预测的标签为：%d,结果为：%d' % (step, pred_y, label))
    except tf.errors.OutOfRangeError:
        print('Done testing -- epoch limit reached')
    print('预测的准确率为：%.02f' % (acc_count / (step + 1)))
    sess.close()


def train(train_dataset,batch_size,n_classes,learning_rate):
    train_iterator = train_dataset.make_initializable_iterator()
    # 训练
    train_x = tf.placeholder(shape=[batch_size,image_width,image_heigh,3],dtype=tf.float32)
    train_y = tf.placeholder(shape=[batch_size],dtype=tf.int32)
    #cnnModel = CnnModel(image_batch[0], imageRes.n_classes)
    cnnModel = CnnNetwork(train_x, n_classes)
    train_logits = cnnModel.create()
    train_loss = loss(train_logits, train_y)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_y)
    # 这个是log汇总记录
    tf.summary.scalar('accuracy', train_acc)
    tf.summary.scalar('loss', train_loss)
    merged_summaries = tf.summary.merge_all()
    
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(train_iterator.make_initializer(train_dataset))
    train_writer = tf.compat.v1.summary.FileWriter(model_path, sess.graph)

    saver = tf.compat.v1.train.Saver() # 产生一个saver来存储训练好的模型

    try:
        for step in range(1, step_count+1):
            images, labels = sess.run(train_iterator.get_next())
            feed_dict = { train_x:images, train_y:labels }
            _, tra_acc, tra_loss = sess.run([train_op, train_acc, train_loss], feed_dict=feed_dict)
            print('%03d 训练损失为：%.6f, 准确率为：%.2f%%' % (step, tra_loss, tra_acc * 100.0))
            if step % 100 == 0:
                summary = sess.run(merged_summaries, feed_dict=feed_dict)
                train_writer.add_summary(summary=summary, global_step=step)
                # 保存训练好的模型
                checkpoint_path = os.path.join(model_path, 'thing.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    sess.close()

if __name__ == '__main__':
    recordRes = RecordRes('./data', image_width=image_width, image_height=image_heigh, batch_size=batch_size)
    recordRes.load_record()

    #train(recordRes.train_dataset,batch_size,recordRes.n_classes,learning_rate)
    print('训练完毕，开始测试。。。。。。')
    test(recordRes.test_dataset, recordRes.n_classes)
