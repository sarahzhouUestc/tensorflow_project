# encoding=utf-8
"""将mnist输入数据转化为TFRecord的格式"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#生成整数列表型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串列表型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist=input_data.read_data_sets('/tmp/data/mnist',dtype=tf.uint8,one_hot=True)
images=mnist.train.images
labels=mnist.train.labels
pixels=images.shape[1]
num_examples=mnist.train.num_examples
record_path='/tmp/data/tfrecord/mnist.tfrecords'
writer=tf.python_io.TFRecordWriter(record_path)
for i in range(num_examples):
    #将矩阵图像转化为一个字符串
    image_raw=images[i].tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
        'pixels':_int64_feature(pixels),
        'label':_int64_feature(np.argmax(labels[i])),
        'image_raw':_bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()