#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: iflyings
import os
import tensorflow as tf
import numpy as np

class ImageRes:
    def __init__(self, path, image_width = 100, image_height = 100, batch_size = 20):
        self.path = path
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.capacity = 0
        self.n_classes = 0

    def load(self):
        categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
        self.n_classes = len(categories)
        image_infos = list()
        for index, category in enumerate(categories):
            image_infos = image_infos + [[index, os.path.join(category, file)] for file in os.listdir(category)]
        np.random.shuffle(image_infos) # 乱序
        self.capacity = len(image_infos)
        
        image_list = [ image_info[1] for image_info in image_infos ]
        label_list = [ image_info[0] for image_info in image_infos ]
        # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
        # tf.cast()用来做类型转换
        image = tf.cast(image_list, tf.string)   # 可变长度的字节数组.每一个张量元素都是一个字节数组
        label = tf.cast(label_list, tf.int32)
        # tf.train.slice_input_producer是一个tensor生成器
        # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
        input_queue = tf.train.slice_input_producer([image, label]) # num_epochs 表示样本循环num_epochs次
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像
    
        # step2：将图像解码，使用相同类型的图像
        image = tf.image.decode_jpeg(image_contents, channels=3)
        # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    
        # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
        image = tf.image.resize_image_with_crop_or_pad(image, self.image_width, self.image_height)
        # 对resize后的图片进行标准化处理
        image = tf.image.per_image_standardization(image)
        
        # step4：生成batch
        # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
        # label_batch: 1D tensor [batch_size], dtype = tf.int32
        image_batch, label_batch = tf.train.batch([image, label], batch_size=self.batch_size, num_threads=6, capacity=self.capacity)
    
        # 重新排列label，行数为[batch_size]
        #label_batch = tf.reshape(label_batch, [batch_size])
        #image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
        image_batch = tf.cast(image_batch, tf.float32)    # 显示灰度图
        return image_batch, label_batch


if __name__ == '__main__':
    imageRes = ImageRes('/home/iflyings/VSCodeProjects/flowers')
    image_batch = imageRes.load()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()# 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)# 启动QueueRunner, 此时文件名队列已经进队
    try:
        step = 0
        # 执行MAX_STEP步的训练，一步一个batch
        while not coord.should_stop():
            step = step + 1
            # 启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
            image, label = sess.run(image_batch)
            #print(image.shape)
            #print(label.shape)
            print('Step %d, image len = %d, label len = %d' % (step, len(image), len(label)))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()# When done, ask the threads to stop.
    coord.join(threads)

    # 保存最后一次网络参数
    sess.close()