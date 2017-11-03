# encoding=utf-8
"""计算5层神经网络带L2正则化的损失函数的计算方法"""
import tensorflow as tf
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
batch_size=8
#定义每层网络的节点个数
#输入节点是2个，输出节点是1个，隐层有3层，每层10个节点
dimensions=[2,10,10,10,1]
layers=len(dimensions)
#当前正在处理的网络层，和对应的维度
in_dimension=dimensions[0]
curr_layer=x #当前输入
#5层全连接神经网络的生成过程/前向传播过程
for i in range(1,layers):
    out_dimension=dimensions[i];
    #生成当前层的权重，并将这个权重变量的L2正则化损失加入到计算图的集合中
    weight=tf.Variable(tf.random_normal([in_dimension,out_dimension]),dtype=tf.float32,trainable=True)
    #加入到当前默认的graph的collection中。
    tf.add_to_collection('loss',tf.contrib.layers.l2_regularizer(0.001)(weight))
    bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]),dtype=tf.float32)
    #使用relu激活函数
    result=tf.matmul(curr_layer,weight)+bias
    curr_layer=tf.nn.relu(result)
    in_dimension=out_dimension

#在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了默认graph的集合中
#这里只需要计算刻画模型在训练数据上的表现的损失函数
#在最后，curr_layer是整个网络的输出
mse_loss=tf.reduce_mean(tf.square(y_-curr_layer))
#将均方误差加入损失函数集合
tf.add_to_collection("loss",mse_loss)

#Graph类中的self._collections维护的是dict，键是str，值是list
loss=tf.add_n(tf.get_collection("loss"))
