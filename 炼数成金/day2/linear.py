# encoding=utf-8
"""使用梯度下降来求解线性模型"""
import tensorflow as tf
import numpy as np
x_data=np.random.rand(100)
y_data=[x*50+2.7+np.random.rand()/1000-0.0005 for x in x_data]

#----------------以下是模型训练的部分------------------
w=tf.Variable(initial_value=np.random.randn(),trainable=True)
b=tf.Variable(initial_value=np.random.randn(),trainable=True)
y=w*x_data+b
loss=tf.reduce_mean(tf.square(y_data-y))
train_step=tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for _ in range(200000):
        if _%100==0:
            print(sess.run([train_step,w,b,loss]))
# print(x_data)
# print([x+np.random.rand()/10-0.05 for x in x_data])