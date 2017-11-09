# encoding=utf-8
"""用来展示滑动平均值的用法"""
import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,trainable=True)
step=tf.Variable(0,dtype=tf.int32,trainable=False)
movingAverage=tf.train.ExponentialMovingAverage(decay=0.99,num_updates=step)
average_op=movingAverage.apply([v])

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(average_op)
    print(sess.run([v,movingAverage.average(v)]))

    sess.run(v.assign(10))
    sess.run(average_op)
    print(sess.run([v,movingAverage.average(v)]))

    sess.run(v.assign(5))
    sess.run(step.assign(10000))
    sess.run(average_op)
    print(sess.run([v,movingAverage.average(v)]))
    sess.run(average_op)
    print(sess.run([v,movingAverage.average(v)]))
    sess.run(average_op)
    print(sess.run([v,movingAverage.average(v)]))
