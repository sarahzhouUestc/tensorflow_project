# encoding=utf-8
import tensorflow as tf
input1=tf.placeholder(tf.float32,shape=[2,3])
input2=tf.placeholder(tf.float32,shape=[2,3])
mul=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(mul,feed_dict={input1:[[1,2,3],[4,5,6]],input2:[[2,3,4],[5,6,7]]}))

