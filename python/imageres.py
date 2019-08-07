#!/home/iflyings/VSCode/venv/tensorflow-venv python
# -*- coding:utf-8 -*-
# Author: iflyings
import os

import numpy as np
import tensorflow as tf

class ImageRes:
    def __init__(self, path, image_width = 100, image_height = 100, batch_size = 20, num_epochs = 100):
        self.path = path
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.capacity = 0
        self.n_classes = 0
        self.train_list = None
        self.test_list = None
        self.__load_image_list()

    def __load_image_list(self):
        categories = [os.path.join(self.path, category) for category in os.listdir(self.path) 
                                        if os.path.isdir(os.path.join(self.path, category))]
        self.n_classes = len(categories)
        image_infos = list()
        for index, category in enumerate(categories):
            image_infos = image_infos + [(index, os.path.join(category, file)) for file in os.listdir(category)]
        self.capacity = len(image_infos)
        np.random.shuffle(image_infos) # 乱序
        train_count = int(self.capacity * 0.9)
        self.train_list = image_infos[:]
        self.test_list = image_infos[train_count:]

    def __load_tf_dataset(self, image_infos, batch_size, num_epochs = None):
        def __parse_dataset(path, label):
            image = tf.io.read_file(path)
            # jpeg或者jpg格式都用decode_jpeg函数
            image = tf.image.decode_jpeg(image, channels=3)
            # 数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
            image = tf.image.resize_with_crop_or_pad(image, self.image_width, self.image_height)
            # 对resize后的图片进行标准化处理
            image = tf.image.per_image_standardization(image)
            #image /= 255.0  # normalize to [0,1] range
            return image, label
        # tf.cast()用来做类型转换
        image_list = tf.cast([image_info[1] for image_info in image_infos], tf.string)
        label_list = tf.cast([image_info[0] for image_info in image_infos], tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        dataset = dataset.map(__parse_dataset)
        dataset = dataset.shuffle(batch_size * 3).batch(batch_size).repeat(num_epochs)
        return dataset

    def load_train_dataset(self):
        return self.__load_tf_dataset(self.train_list, self.batch_size)

    def load_test_dataset(self):
        return self.__load_tf_dataset(self.test_list, 1, num_epochs = 1)

if __name__ == '__main__':
    #imageRes = ImageRes('./data')
    #imageRes.load_dataset()
    #train_dataset = imageRes.train_dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(np.arange(1,7))
    train_dataset = train_dataset.shuffle(4).repeat().batch(2)
    train_iterator = train_dataset.make_initializable_iterator()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.make_initializer(train_dataset))

    next_element = train_iterator.get_next()
    try:
        for step in range(100):
            labels = sess.run(next_element)
            print('Step %04d, label = [%d %d]' % (step, labels[0], labels[1]))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    # 保存最后一次网络参数
    sess.close()
