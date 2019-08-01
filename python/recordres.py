#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: iflyings
import os
import glob
import tensorflow as tf
import numpy as np
from PIL import Image

class RecordRes:
    def __init__(self, path, image_width = 100, image_height = 100, batch_size = 20):
        self.path = path
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.capacity = 0

    def __create(self, tfrecords_path):
        if not os.path.exists(tfrecords_path):
            categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
            self.n_classes = len(categories)
            image_infos = list()
            for index, category in enumerate(categories):
                image_infos = image_infos + [[index, os.path.join(category, file)] for file in os.listdir(category)]
            np.random.shuffle(image_infos)
            self.capacity = len(image_infos)

            writer = tf.python_io.TFRecordWriter(tfrecords_path)
            for image_info in image_infos:
                image = Image.open(image_info[1])
                image = np.array(image.resize((self.image_width, self.image_height)))
                image = image.tobytes()
                example = tf.train.Example(features = tf.train.Features(feature = 
                    {
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_info[0]])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                    }))
                writer.write(example.SerializeToString())
            writer.close()


    def load(self):
        tfrecords_path = glob.glob(os.path.join(self.path, '*.tfrecords'))
        if not tfrecords_path:
            tfrecords_path = os.path.join(self.path, 'record_train.tfrecords')
            self.__create(tfrecords_path)
        else:
            categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
            self.n_classes = len(categories)
            image_infos = list()
            for index, category in enumerate(categories):
                image_infos = image_infos + [[index, os.path.join(category, file)] for file in os.listdir(category)]
            self.capacity = len(image_infos)
        filename_queue = tf.train.string_input_producer(tfrecords_path, num_epochs=3, capacity=self.capacity)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue) 
        features = tf.parse_single_example(serialized_example, features = 
            {
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
            })
        label_batch = features['label']
        label_batch = tf.cast(label_batch, tf.int64)
        image_batch = tf.decode_raw(features['img_raw'], tf.uint8)
        image_batch = tf.reshape(image_batch, [self.image_width, self.image_height, 3])
        image_batch = tf.cast(image_batch, tf.float32) * (1. / 255) - 0.5
        return tf.train.batch([image_batch, label_batch], self.batch_size, capacity = self.capacity) # 批量取出


if __name__ == '__main__':
    imageRes = RecordRes('/home/iflyings/VSCodeProjects/flowers')
    image_batch = imageRes.load()
    # print(image_batch[0].shape)
    # print(image_batch[1].shape)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()# 创建一个线程协调器，用来管理之后在Session中启动的所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)# 启动QueueRunner, 此时文件名队列已经进队
    try:
        # 执行MAX_STEP步的训练，一步一个batch
        step = 0
        while not coord.should_stop():
            step = step + 1
            image, label = sess.run(image_batch)
            print('Step %d, image len = %d, label len = %d' % (step, len(image), len(label)))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()# When done, ask the threads to stop.
    coord.join(threads)

    sess.close()
