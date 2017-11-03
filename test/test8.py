# encoding=utf-8
"""训练神经网络解决二分类问题"""
import tensorflow as tf
from numpy.random import RandomState
batch_size=8
#定义神经网络的参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.placeholder(tf.float32,shape=[None,2],name='x-input')
y_=tf.placeholder(tf.float32,shape=[None,1],name='y-input') #labels

#前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
#定义损失函数和反向传播的算法
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm=RandomState(1)
data_size=128
#total samples
X=rdm.rand(data_size,2)
print(type(X))
print(X)
#total labels
Y=[[int((x1+x2)<1)] for (x1,x2) in X]

STEPS=5000

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start=(i*batch_size)%data_size
        end=min(start+batch_size,data_size)
        #使用选取的样本训练神经网络来更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        #每隔一段时间计算所有数据上的交叉熵并输出日志
        if i%1000==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s), cross entropy on all data is %g"%(i,total_cross_entropy))
            #交叉熵越小，说明预测的结果和真实的结果差距越小
    #训练完毕后的参数
    print(w1.eval(session=sess))
    print(w2.eval(session=sess))

    #训练完毕后的总交叉熵
    final_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
    print("The last loss is %f"%total_cross_entropy)

