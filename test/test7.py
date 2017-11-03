# encoding=utf-8
import tensorflow as tf

a=tf.random_normal([2,3])
b=tf.random_normal([2,4])
sess=tf.Session()
print(a.eval(session=sess))
print(b.eval(session=sess))

print(a.eval(session=sess))
print(b.eval(session=sess))

sess.close()
