# encoding=utf-8
"""
采用上一章中提到的进一步优化方法
1.指数衰减学习率
2.正则化避免过度拟合
3.滑动平均使模型更健壮
只有一个隐藏层hidden layer
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
input_node=784
output_node=10
layer_node=500
batch_size=100
#---------学习率---------
learning_rate=0.8   #基础的学习率
decay_rate=0.99     #学习率的衰减率
#learning_rate=learning_rate*decay_rate^(global_step/decay_steps)

#---------正则项---------
regularization_rate=0.0001

#---------滑动平均---------
moving_average_decay=0.99

steps=30000

def inference(input_tensor,expoMovingAverage,w1,b1,w2,b2):
    if expoMovingAverage == None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
        return tf.matmul(layer1,w2)+b2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,expoMovingAverage.average(w1))+expoMovingAverage.average(b1))
        return tf.matmul(layer1,expoMovingAverage.average(w2))+expoMovingAverage.average(b2)

def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,input_node],name="x-input")
    y_=tf.placeholder(tf.float32,shape=[None,output_node],name="y-input")
    weights1=tf.Variable(tf.truncated_normal(shape=[input_node,layer_node],stddev=0.1),trainable=True)
    biases1=tf.Variable(tf.constant(0.1,shape=[layer_node]))
    weights2=tf.Variable(tf.truncated_normal(shape=[layer_node,output_node],stddev=0.1),trainable=True)
    biases2=tf.Variable(tf.constant(0.1,shape=[output_node]))
    #没使用滑动平均前向预测的值
    y=inference(x,None,weights1,biases1,weights2,biases2)

    #-----------以下将使用滑动平均------------
    global_step=tf.Variable(0,trainable=False)
    ema=tf.train.ExponentialMovingAverage(decay=moving_average_decay,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    y_average=inference(x,ema,weights1,biases1,weights2,biases2)

    #非滑动平均值的结果与标准答案的交叉熵
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.arg_max(y_,axis=1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    #-----------以下将使用L2正则化------------
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    loss=cross_entropy_mean+regularizer(weights1)+regularizer(weights2)

    #-----------以下将使用指数衰减学习率----------
    learning=tf.train.exponential_decay(learning_rate=learning_rate,global_step=global_step,\
            decay_steps=mnist.train.num_examples/batch_size,decay_rate=decay_rate)

    train_step=tf.train.GradientDescentOptimizer(learning).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op("train")  #no_op没有实际作用

    #检验使用了滑动平均模型的预测结果是否正确，得到长度为batch_size的bool向量
    correct_prediction=tf.equal(tf.argmax(y_average,axis=1),tf.argmax(y_,axis=1))
    accuracy_average=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        for i in range(steps):
            if i%1000==0:
                validation_acc=sess.run(accuracy_average,feed_dict=validation_feed)
                print("After %d training step(s), validation accuracy using average model is %g") % (i,validation_acc)
            xt,yt=mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:xt,y_:yt})
        #训练结束后，使用测试集得出最终的正确率
        test_acc=sess.run(accuracy_average,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (steps,test_acc))

    def main(argv=None):
        mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
        train(mnist)

    # tf.app.run()是tensorflow提供的主程序入口，会调用上面定义的main函数
    if __name__ == '__main__':
        tf.app.run()