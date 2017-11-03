# encoding=utf-8
"""测试tf.add_to_colletion方法"""
import tensorflow as tf

tf.add_to_collection("num",100)
tf.add_to_collection("num",1000)
tf.add_to_collection("num",10000)
tf.add_to_collection("name","sarah")
tf.add_to_collection("name","zhou")

print(type(tf.get_collection("num")))
print(tf.get_collection("num"))
print(tf.get_collection("name"))