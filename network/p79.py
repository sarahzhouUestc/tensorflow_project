# encoding=utf-8
"""
p79页的自定义损失函数实现
batch_size=8
没有隐藏层,只有一组weight权重向量
使用RandomState生成数据集
在标准label中加入噪音
回归问题，不是分类问题

batch_size=8
输入placeholder
输出placeholder
损失函数：自定义
优化器：AdamOptimizer
"""
import tensorflow as tf
from numpy.random import RandomState
BATCH_SIZE=8
LOSS_MORE=1
LOSS_LESS=10
x=tf.placeholder(tf.float32,(None,2),"x_input")
y_=tf.placeholder(tf.float32,(None,1),"y_input")
weights=tf.Variable(tf.random_normal([2,1],mean=0,stddev=1,seed=1,dtype=tf.float32),trainable=True)
y=tf.matmul(x,weights)
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*LOSS_MORE,(y_-y)*LOSS_LESS))

#优化过程，learning rate使用固定的学习率
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
#=================以上是定义tensorflow图的过程=================
DATASET_SIZE=128
rdm=RandomState(1)
X=rdm.rand(DATASET_SIZE,2) #生成均匀分布uniform distribution
Y_=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]
with tf.Session() as sess:
    init_op=tf.global_variables_initializer();
    sess.run(init_op)   #初始化变量Variable
    STEPS=5000  #运行5000个batch
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%DATASET_SIZE
        end=min(start+BATCH_SIZE,DATASET_SIZE)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})

    print(weights.eval())





