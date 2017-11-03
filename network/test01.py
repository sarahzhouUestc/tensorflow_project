# encoding=utf-8
"""这是自定义损失函数，有点像风险损失函数"""

import tensorflow as tf
from numpy.random import RandomState
batch_size=8

x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")
# correct output, not prediction
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-output")
w=tf.Variable(initial_value=tf.random_normal([2,1],seed=1),trainable=True)
# prediction
y=tf.matmul(x,w)

loss_less=10
loss_more=1
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)
# 以上定义了运行模型，graph

rdm=RandomState(1) #1是种子，保证每次运行结果都一样
data_size=128
# 样本
X=rdm.rand(data_size,2)
# 来设置正确的output，同时加上随机量
# rdm.rand()的结果是[0,1)的均匀分布中的随机数
Y=[[x1+x2+rdm.rand()/10-0.05] for (x1,x2) in X]
sess=tf.InteractiveSession();
init_op=tf.global_variables_initializer();
sess.run(init_op)
print(w.eval())
# print(sess.run(loss))

EPOCH=5000
for i in range(EPOCH):
    start=(i*batch_size)%data_size
    end=min(start+batch_size,data_size)
    sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

print(w.eval())
# print(sess.run(loss))
