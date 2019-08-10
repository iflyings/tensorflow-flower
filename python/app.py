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
from vgg16 import VGG16

image_width = 224
image_heigh = 224
batch_size = 256
learning_rate = 0.01
step_count = 50000
image_path = './flowers'
model_path = './model'

def loss(logits, label_batches):
    with tf.compat.v1.variable_scope('loss') as scope:
        # 使用交叉熵损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_batches, name='softmax')
        return tf.reduce_mean(cross_entropy, name='loss')

def trainning(loss, learning_rate):
    #update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    return tf.compat.v1.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)

def evaluation(logits, label_batches):
    with tf.compat.v1.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, label_batches, 1)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

def test(test_dataset, n_classes):
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
    next_element = test_iterator.get_next()

    cnn_model = CnnNetwork(next_element[0], 1, n_classes, False)
    test_logits = cnn_model.create()
    test_logits = tf.reshape(test_logits, [n_classes])
    test_predict = tf.nn.softmax(test_logits)
    test_label = next_element[1]

    config = tf.ConfigProto(device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
                    inter_op_parallelism_threads = 1, 
                    intra_op_parallelism_threads = 4,
                    log_device_placement=True)
    sess = tf.compat.v1.InteractiveSession(config = config)
    sess.run(tf.compat.v1.global_variables_initializer())
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
            pred_y, label = sess.run([test_predict, test_label])
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
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    next_element = train_iterator.get_next()
    # 训练
    #train_x = tf.placeholder(shape=[batch_size,image_width,image_heigh,3],dtype=tf.float32)
    #train_y = tf.placeholder(shape=[batch_size],dtype=tf.int32)
    #cnnModel = CnnModel(image_batch[0], imageRes.n_classes)
    cnn_model = VGG16(n_classes, True)
    train_logits = cnn_model.create(next_element[0])
    train_loss = loss(train_logits, next_element[1])
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, next_element[1])
    # 这个是log汇总记录
    tf.compat.v1.summary.scalar('accuracy', train_acc)
    tf.compat.v1.summary.scalar('loss', train_loss)
    merged_summaries = tf.compat.v1.summary.merge_all()
    # 产生一个saver来存储训练好的模型
    saver = tf.compat.v1.train.Saver()

    config = tf.ConfigProto(device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
                    inter_op_parallelism_threads = 8, 
                    intra_op_parallelism_threads = 8,
                    log_device_placement=True)
    sess = tf.compat.v1.InteractiveSession(config = config)
    sess.run(tf.compat.v1.global_variables_initializer())

    train_writer = tf.compat.v1.summary.FileWriter(model_path, sess.graph)
    try:
        for step in range(1, step_count+1):
            _, tra_acc, tra_loss = sess.run([train_op, train_acc, train_loss])
            print('%03d 训练损失为：%.5f, 准确率为：%.2f%%' % (step, tra_loss, tra_acc * 100.0))
            if step % 100 == 0:
                summary = sess.run(merged_summaries)
                train_writer.add_summary(summary=summary, global_step=step)
                # 保存训练好的模型
                checkpoint_path = os.path.join(model_path, 'thing.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    sess.close()

if __name__ == '__main__':
    imageRes = ImageRes(image_path, image_width=image_width, image_height=image_heigh, batch_size=batch_size)

    tf.reset_default_graph()
    train(imageRes.load_train_dataset(),batch_size,imageRes.n_classes,learning_rate)
    print('训练完毕，开始测试。。。。。。')
    tf.reset_default_graph()
    test(imageRes.load_test_dataset(), imageRes.n_classes)
