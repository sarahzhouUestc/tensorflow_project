# encoding=utf-8
import tensorflow as tf

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
print(tf.global_variables())
print(tf.trainable_variables())

a=tf.Variable(tf.random_normal([1,3],stddev=2,dtype=tf.float32),name='a')
b=tf.Variable(tf.random_normal([2,3],stddev=1,dtype=tf.float32),name='b')
tf.assign(a,b,validate_shape=False)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

print("=============================see variable value================================")
print(a.eval(session=sess))
print(sess.run(b))

sess.close()

