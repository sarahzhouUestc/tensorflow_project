# encoding=utf-8
"""将mnist数据存储到TFRecord文件中"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets("/tmp/data/mnist",dtype=tf.uint8,one_hot=True)
images=mnist.train.images
pixel=images.shape[1]
labels=np.argmax(mnist.train.labels,axis=1)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

record_filename="/tmp/data/tfrecord/mnist.tfrecords"
writer=tf.python_io.TFRecordWriter(record_filename)
for i in range(mnist.train.num_examples):
    image_raw=images[i].tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
        'pixel':_int64_feature(pixel),
        'label':_int64_feature(labels[i]),
        'image_raw':_bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()  