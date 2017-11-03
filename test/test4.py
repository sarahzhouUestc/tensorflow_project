# encoding=utf-8

import tensorflow as tf
#使用seed随机种子，可以保证每次运行得到的结果是一样的
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
print(w1)
print(w2)
x=tf.constant([[0.7,0.9]])

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
print(type(y))

print("===========================before run========================")
print(w1)
print(w1.op)
print("===========================run operation========================")
sess=tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(w1)
print(w1.op)

print(sess.run(y))
print(type(y))
print(dir(y))
print(y.shape)
print(y.eval(session=sess))
sess.close()
