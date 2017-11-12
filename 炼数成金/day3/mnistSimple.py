# encoding=utf-8
"""简单的mnist数字识别实现"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
batch_size=64
batchs=mnist.train.num_examples//batch_size+1

#占位符
x_data=tf.placeholder(shape=[None,784],name="x-input",dtype=tf.float32)
y_data=tf.placeholder(shape=[None,10],name="y-input",dtype=tf.float32)

#没有隐藏层，只有输入层和输出层
w=tf.Variable(tf.random_normal([784,10]),name="weight")
b=tf.Variable(tf.random_normal([10]),name="biases")
#采用softmax
prediction=tf.nn.softmax(tf.matmul(x_data,w)+b)
#损失函数采用误差平方和
loss=tf.reduce_mean(tf.square(y_data-prediction))
#优化器采用梯度下降法optimizer
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#正确率
correct_bool=tf.equal(tf.argmax(y_data,axis=1),tf.argmax(prediction,axis=1))
acc=tf.reduce_mean(tf.cast(correct_bool,tf.float32))

epochs=1000
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(epochs):
        for j in range(batchs):
            xdata,xlabels=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x_data:xdata,y_data:xlabels})

    print(sess.run(acc,feed_dict={x_data:mnist.train.images,y_data:mnist.train.labels}))
    print(sess.run(acc,feed_dict={x_data:mnist.test.images,y_data:mnist.test.labels}))




