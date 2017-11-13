# encoding=utf-8
"""比较使用滑动平均模型和不使用滑动平均模型"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE=64
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
learning_rate_base=0.8
learning_decay_rate=0.99
decay_steps=mnist.train.num_examples//BATCH_SIZE+1
#learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
regularization_rate=0.0001
moving_decay_rate=0.99
input_node=784
output_node=10
hidden1_node=100
hidden2_node=100
EPOCHS=1000

def inference(x,w1,b1,w2,b2,w3,b3,ema):
    if ema==None:
        hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
        hidden2=tf.nn.relu(tf.matmul(hidden1,w2)+b2)
        return tf.matmul(hidden2,w3)+b3
    else:
        hidden1=tf.nn.relu(tf.matmul(x,ema.average(w1))+ema.average(b1))
        hidden2=tf.nn.relu(tf.matmul(hidden1,ema.average(w2))+ema.average(b2))
        return tf.matmul(hidden2,ema.average(w3))+ema.average(b3)

def train(mnist):
    x=tf.placeholder(tf.float32,[None,input_node],name="x-input")
    y_=tf.placeholder(tf.float32,[None,output_node],name="y-input")
    w1=tf.Variable(tf.truncated_normal([input_node,hidden1_node],stddev=0.1),dtype=tf.float32)
    b1=tf.Variable(tf.constant(0.1,shape=[hidden1_node]),dtype=tf.float32)
    w2=tf.Variable(tf.truncated_normal([hidden1_node,hidden2_node],stddev=0.1),dtype=tf.float32)
    b2=tf.Variable(tf.constant(0.1,shape=[hidden2_node]),dtype=tf.float32)
    w3=tf.Variable(tf.truncated_normal([hidden2_node,output_node],stddev=0.1),dtype=tf.float32)
    b3=tf.Variable(tf.constant(0.1,shape=[output_node]),dtype=tf.float32)
    global_step=tf.Variable(0,trainable=False)
    ema=tf.train.ExponentialMovingAverage(moving_decay_rate,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    #L2正则化项
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    y=inference(x,w1,b1,w2,b2,w3,b3,None)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,axis=1),logits=y)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+regularizer(w1)+regularizer(w2)+regularizer(w3)
    #指数衰减学习率
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,decay_steps,learning_decay_rate)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    with tf.control_dependencies([ema_op,train_step]):
        train_op=tf.no_op("train")
    #accuracy
    correct_bool=tf.equal(tf.argmax(y_,axis=1),tf.argmax(y,axis=1))
    accuracy=tf.reduce_mean(tf.cast(correct_bool,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        for _ in range(EPOCHS):
            for i in range(decay_steps):
                step=sess.run(global_step)
                xs,ys=mnist.train.next_batch(BATCH_SIZE)
                if step%1000==0:
                    validation_acc = sess.run(accuracy, feed_dict=validation_feed)
                    print("After %d step(s),the acc on validation is %f" % (step, validation_acc))
                    train_acc = sess.run(accuracy, feed_dict={x:xs,y_:ys})
                    print("After %d step(s),the acc on train is %f" % (step, train_acc))
                sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("After %d step(s),the acc on test is %f" % (step,test_acc))

def main(argv=None):
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
