# encoding=utf-8
"""从TFRecord中读取数据"""
import tensorflow as tf
#创建一个reader来读取TFRecord
reader=tf.TFRecordReader()
#创建一个队列来维护输入文件列表
file_queue=tf.train.string_input_producer(["/tmp/data/tfrecord/mnist.tfrecords"])
_,serialized_example=reader.read(file_queue)

features=tf.parse_single_example(serialized_example,
                                 features={
                                     'image_raw':tf.FixedLenFeature([],tf.string),
                                     'label':tf.FixedLenFeature([],tf.int64),
                                     'pixel':tf.FixedLenFeature([],tf.int64)
                                 })
images=tf.decode_raw(features['image_raw'],tf.uint8)
labels=tf.cast(features['label'],tf.int32)
pixels=tf.cast(features['pixel'],tf.int32)

sess=tf.Session()
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    image,label,pixel=sess.run([images,labels,pixels])
    print(image.shape)
    print(label)
    print(pixel)

