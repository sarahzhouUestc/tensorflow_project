# encoding=utf-8
"""演示变量自增的操作"""
import tensorflow as tf

counter=tf.Variable(0)
new_value=tf.add(counter,1)
update=tf.assign(counter,new_value)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer();
    sess.run(init_op)
    for _ in range(5):
        sess.run(update)
        print(sess.run(counter))