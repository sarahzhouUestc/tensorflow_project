# encoding=utf-8
"""从TFRecord文件中读取数据"""
import tensorflow as tf
reader=tf.TFRecordReader()
#创建一个队列来维护输入文件列表
filename_queue=tf.train.string_input_producer(['/tmp/data/tfrecord/mnist.tfrecords'])
#从文件中读取一个样例
_,serialized_example=reader.read(filename_queue)
#解析读出的样例
features=tf.parse_single_example(serialized_example,
                                 features={
                                     'image_raw':tf.FixedLenFeature([],tf.string),
                                     'pixel':tf.FixedLenFeature([],tf.int64),
                                     'label':tf.FixedLenFeature([],tf.int64)
                                 })
#解析字符串为图像对应的像素数组
images=tf.decode_raw(features['image_raw'],tf.uint8)
labels=tf.cast(features['label'],tf.int32)
pixels=tf.cast(features['pixel'],tf.int32)

sess=tf.Session()
#启动多线程处理数据
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    image,label,pixel=sess.run([images,labels,pixels])