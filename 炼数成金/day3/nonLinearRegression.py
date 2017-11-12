# encoding=utf-8
"""使用梯度下降法来做非线性回归"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#--------------以下是构造数据的部分--------------
x_data=np.linspace(-1,1,num=400,dtype=float)[:,np.newaxis]
noise=np.random.normal(0,0.02,size=x_data.shape)
y_data=np.square(x_data)+noise


#--------------以下是模型训练部分---------------
x=tf.placeholder(shape=[None,1],dtype=tf.float32)
w1=tf.Variable(initial_value=tf.random_normal(shape=[1,10]),trainable=True)
b1=tf.Variable(initial_value=tf.constant(0.,shape=[10]),dtype=tf.float32)
w2=tf.Variable(initial_value=tf.random_normal(shape=[10,1]),trainable=True)
b2=tf.Variable(initial_value=tf.constant(0.,shape=[1]),dtype=tf.float32)
hidden_layer=tf.nn.tanh(tf.matmul(x,w1)+b1)
output=tf.nn.tanh(tf.matmul(hidden_layer,w2)+b2)

loss=tf.reduce_mean(tf.square(y_data-output))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for _ in range(600000):
        sess.run(train_step,feed_dict={x:x_data})
        if _%100==0:
            print("After %d step(s), the loss is %g" % (_,sess.run(loss,feed_dict={x:x_data})))

    #--------------以下是数据图形显示部分---------------
    plt.figure(figsize=(25,16))
    plt.scatter(x_data,y_data,s=5)
    plt.plot(x_data,sess.run(output,feed_dict={x:x_data}),'r')
    plt.show()



