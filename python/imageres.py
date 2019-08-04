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
        self.train_dataset = None
        self.test_dataset = None

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
        dataset = dataset.shuffle(batch_size * 3).repeat(num_epochs).batch(batch_size)
        return dataset

    def load_dataset(self):
        categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
        self.n_classes = len(categories)
        image_infos = list()
        for index, category in enumerate(categories):
            image_infos = image_infos + [(index, os.path.join(category, file)) for file in os.listdir(category)]
        self.capacity = len(image_infos)
        np.random.shuffle(image_infos) # 乱序
        
        train_count = int(self.capacity * 0.9)
        self.train_dataset = self.__load_tf_dataset(image_infos[:train_count],self.batch_size)
        self.test_dataset = self.__load_tf_dataset(image_infos[train_count:],1,num_epochs = 1)


if __name__ == '__main__':
    step_count = 1000

    imageRes = ImageRes('./data')
    imageRes.load_dataset()
    train_dataset = imageRes.train_dataset
    train_iterator = train_dataset.make_initializable_iterator()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.make_initializer(train_dataset))

    try:
        for step in range(step_count):
            images, labels = sess.run(train_iterator.get_next())
            print('Step %04d, image len = %d, label len = %d' % (step, len(images), len(labels)))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    # 保存最后一次网络参数
    sess.close()
