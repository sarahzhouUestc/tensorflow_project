# encoding=utf-8
"""mnist手写数字识别升级版"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=100
EPOCHS=1000

#-----------指数衰减学习率-----------
learning_rate_base=0.8
learning_decay_rate=0.99
#-----------learning_rate=learning_rate*decay_rate^(global_step/decay_steps)

#-----------滑动平均值---------------
moving_decay=0.99
#-----------shadow_variable=decay*shadow_variable+(1-decay)*variable-------------
#-----------decay=min(decay,(1+num_updates)/(10+num_updates))------------

#-----------正则化------------------
regularization_rate=0.0001
#-----------r(w)=λR(w)

def inference(x,w1,b1,w2,b2,w3,b3,ema):
# def inference(x,w1,b1,w2,b2,ema):
    if ema==None:
        hidden1=tf.nn.relu(tf.matmul(x, w1) + b1)
        hidden2=tf.nn.relu(tf.matmul(hidden1, w2) + b2)
        y=tf.matmul(hidden2, w3) + b3
        return y
        # return tf.matmul(hidden1,w2)+b2
    else:
        hidden1=tf.nn.relu(tf.matmul(x,ema.average(w1))+ema.average(b1))
        # return tf.matmul(hidden1,ema.average(w2))+ema.average(b2)
        hidden2=tf.nn.relu(tf.matmul(hidden1,ema.average(w2))+ema.average(b2))
        y=tf.matmul(hidden2,ema.average(w3))+ema.average(b3)
        return y

#-----------以下是模型训练的部分-----------
def train(mnist):
    data_size=mnist.train.num_examples
    global_step=tf.Variable(0,trainable=False)
    x_data=tf.placeholder(tf.float32,shape=[None,784],name="x-input")
    y_data=tf.placeholder(tf.float32,shape=[None,10],name="y-input")
    w1=tf.Variable(tf.truncated_normal([784,100],stddev=0.1),dtype=tf.float32,trainable=True)
    b1=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[100]),trainable=True)
    w2=tf.Variable(tf.truncated_normal([100,100],stddev=0.1),dtype=tf.float32,trainable=True)
    b2=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[100]),trainable=True)
    w3=tf.Variable(tf.truncated_normal([100,10],stddev=0.1),dtype=tf.float32,trainable=True)
    b3=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[10]),trainable=True)

    #滑动平均值
    ema=tf.train.ExponentialMovingAverage(moving_decay,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    #预测输出，使用了滑动平均值
    # y=inference(x_data,w1,b1,w2,b2,w3,b3,ema)
    y=inference(x_data,w1,b1,w2,b2,w3,b3,None)

    #学习率
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,\
            decay_steps=data_size/BATCH_SIZE,decay_rate=learning_decay_rate)
    #交叉熵损失
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_data,axis=1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #L2正则化
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    # loss=cross_entropy_mean+regularizer(w1)+regularizer(w2)+regularizer(w3)
    loss=cross_entropy_mean+regularizer(w1)+regularizer(w2)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([ema_op,train_step]):
        train_op=tf.no_op("train")

    #正确率
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_data, axis=1))
    accuracy_average = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    # prediction_bool=tf.equal(tf.argmax(y_data,axis=1),tf.argmax(y,axis=1))
    # acc=tf.reduce_mean(tf.cast(prediction_bool,dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validation_feed={x_data:mnist.validation.images,y_data:mnist.validation.labels}
        test_feed={x_data:mnist.test.images,y_data:mnist.test.labels}
        for _ in range(EPOCHS):
            for i in range(data_size//BATCH_SIZE+1):
                if sess.run(global_step)%1000==0:
                    print("After %d step(s), the acc on validation is %g" % (sess.run(global_step),sess.run(accuracy_average,feed_dict=validation_feed)))
                    #sess.run([global_step,acc],feed_dict=validation_feed)
                xs,ys=mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op,feed_dict={x_data:xs,y_data:ys})
            print("The acc on train is %g" % sess.run(accuracy_average,feed_dict={x_data:mnist.train.images,y_data:mnist.train.labels}))
        #------在最终测试集上的表现--------
        print("After %d step(s), the acc on test is %g" % (sess.run(global_step),sess.run(accuracy_average,feed_dict=test_feed)))

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
