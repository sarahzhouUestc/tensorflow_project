# encoding=utf-8
"""
演示模型持久化使用的 protocol buffer 数据格式
"""
import tensorflow as tf
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result=v1+v2
saver=tf.train.Saver()
#持久化的第一个文件
saver.export_meta_graph(filename="/tmp/model/metatest/metatest.ckpt.meta.json",as_text=True)