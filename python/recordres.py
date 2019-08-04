#!/home/iflyings/VSCode/venv/tensorflow-venv python
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
        self.train_dataset = None
        self.test_dataset = None

    def __create_tf_record(self, image_infos, tfrecords_path):
        writer = tf.python_io.TFRecordWriter(tfrecords_path)
        for image_info in image_infos:
            label = image_info[0]
            image = Image.open(image_info[1])
            image = np.array(image.resize((self.image_width, self.image_height)))
            image = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature = {
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))
            writer.write(example.SerializeToString())
        writer.close()

    def __load_tf_record(self, tfrecords_path, batch_size=1, num_epochs=None):
        def __parse_dataset(record):
            keys_to_features = {
                'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            image = tf.decode_raw(features['image'], tf.uint8)
            image = tf.reshape(image, [self.image_width, self.image_height, 3])
            #image = tf.cast(image,tf.float32)*(1./255)-0.5
            image = tf.image.per_image_standardization(image)
            label = tf.cast(features['label'], tf.int64)
            return image, label

        dataset = tf.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.map(__parse_dataset)
        dataset = dataset.shuffle(batch_size*5).batch(batch_size,drop_remainder=True).repeat(num_epochs)
        return dataset

    def load_record(self):
        tfrecords_path = glob.glob(os.path.join(self.path, '*.tfrecords'))
        if not tfrecords_path:
            categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
            self.n_classes = len(categories)
            image_infos = list()
            for index, category in enumerate(categories):
                image_infos = image_infos + [(index, os.path.join(category, file)) for file in os.listdir(category)]
            np.random.shuffle(image_infos)
            self.capacity = len(image_infos)

            train_count = int(self.capacity * 0.9)
            self.__create_tf_record(image_infos[:train_count], os.path.join(self.path, 'record_train.tfrecords'))
            self.__create_tf_record(image_infos[train_count:], os.path.join(self.path, 'record_test.tfrecords'))
        else:
            categories = [os.path.join(self.path, category) for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category))]
            self.n_classes = len(categories)
            image_infos = list()
            for index, category in enumerate(categories):
                image_infos = image_infos + [[index, os.path.join(category, file)] for file in os.listdir(category)]
            self.capacity = len(image_infos)

        self.train_dataset = self.__load_tf_record(os.path.join(self.path, 'record_train.tfrecords'), 
                                    batch_size=self.batch_size)
        self.test_dataset = self.__load_tf_record(os.path.join(self.path, 'record_test.tfrecords'),
                                    num_epochs=1)

if __name__ == '__main__':
    recordRes = RecordRes('./data')
    recordRes.load_record()

    train_dataset = recordRes.train_dataset
    train_iterator = train_dataset.make_initializable_iterator()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.make_initializer(train_dataset))

    try:
        for step in range(1000):
            images, labels = sess.run(train_iterator.get_next())
            print('Step %d, image len = %d, label len = %d' % (step, len(images), len(labels)))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    sess.close()
