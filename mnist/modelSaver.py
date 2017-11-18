# encoding=utf-8
"""
tf.train.Saver
saver.save
saver.restore
"""
import tensorflow as tf
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result=v1+v2
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess,"/tmp/model/dict/test.ckpt")

#==================加载模型方式一，这种方式会重复定义图上的运算=====================
# with tf.Session() as sess:
#     saver.restore(sess,"/tmp/model/test/test.ckpt")
#     print(sess.run(result))

#==================加载模型方式二=====================
#test.ckpt.meta中是图结构，直接加载持久化的图，不用重复定义图上的运算，就不需要一开始的变量和运算定义了
# saver=tf.train.import_meta_graph("/tmp/model/test/test.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess,"/tmp/model/test/test.ckpt")
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

#==================加载模型方式三=====================
#保存或加载部分变量，提供列表来指定需要保存或加载的变量
# saver=tf.train.Saver(var_list=[v1])

#==================加载模型方式四=====================
#加载时重命名
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")
saver=tf.train.Saver({"v1":v1,"v2":v2})
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"/tmp/model/dict/test.ckpt")
    print(sess.run(v1))

