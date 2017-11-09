# encoding=utf-8
"""
回归问题
这个模块用来随机生成一个5层的全连接神经网络
相当于一个前向传播的操作
在这里使用了优化方法：损失函数加入L2正则
只是定义了网络结构
没有训练和优化，因此还没有用到指数型学习率
还没有用到实际的训练集来训练
仅仅是定义了5层神经网络的Graph
"""
import tensorflow as tf
from numpy.random import RandomState

#生成指定shape的参数，同时把参数的L2范数加到Graph的Collections中
def get_weights(shape,lam):
    weights=tf.Variable(tf.random_normal(shape),dtype=tf.float32,trainable=True)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lam)(weights))
    return weights

dimensions=[2,10,10,10,1]
layers=len(dimensions)
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y=tf.placeholder(tf.float32,shape=(None,1),name="y-input")
current_in=x
dimension_in=dimensions[0]

for i in range(1,layers):
    dimension_out=dimensions[i]
    w=get_weights([dimension_in,dimension_out],0.01)
    biase=tf.Variable(tf.constant(0.1,shape=[dimension_out]))
    # print(biase)
    # print(tf.matmul(current_in,w).shape)
    current_in=tf.nn.relu(tf.matmul(current_in,w)+biase)
    dimension_in=dimension_out
mse=tf.reduce_mean(tf.square(y-current_in))
tf.add_to_collection("losses",mse)

#生成测试数据
rdm=RandomState(1)
X=rdm.randn(100,2)
Y=rdm.randint(5,10,[100,1])
with tf.Session() as sess:
    init_op=tf.global_variables_initializer();
    sess.run(init_op)
    print(tf.add_n(tf.get_collection("losses")).eval(session=sess,feed_dict={x:X,y:Y}))



