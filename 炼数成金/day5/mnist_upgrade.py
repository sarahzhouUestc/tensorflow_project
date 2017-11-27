# encoding=utf-8
"""
将mnist的分类准确率提高到98%以上
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train(mnist):
    batch_size=100
    n_batch=mnist.train.num_examples//batch_size

    #定义变量
    x=tf.placeholder(tf.float32,[None,784],name="x-input")
    y_=tf.placeholder(tf.float32,[None,10],name="y-input")
    keep_prob=tf.placeholder(tf.float32)
    lr=tf.Variable(0.001,dtype=tf.float32)

    #hidden_layer1
    weights1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1),dtype=tf.float32)
    biase1=tf.Variable(tf.constant(0.0,tf.float32,[500]))
    L1=tf.nn.tanh(tf.matmul(x,weights1)+biase1)
    L1_drop=tf.nn.dropout(L1,keep_prob)

    #hidden_layer2
    weights2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1),dtype=tf.float32)
    biase2=tf.Variable(tf.constant(0.0,tf.float32,[300]))
    L2=tf.nn.tanh(tf.matmul(L1_drop,weights2)+biase2)
    L2_drop=tf.nn.dropout(L2,keep_prob)

    #hidden_layer3
    weight3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1),dtype=tf.float32)
    biase3=tf.Variable(tf.constant(0.0,tf.float32,[10]))
    y=tf.matmul(L2_drop,weight3)+biase3

    #交叉熵损失函数
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    loss=tf.reduce_mean(cross_entropy,axis=0)

    #准确率
    correct_prediction=tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),axis=0)

    train_step=tf.train.AdamOptimizer(lr).minimize(loss)
    init_op=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(50):
            sess.run(tf.assign(lr,0.001*(0.95**epoch)))
            for batch in range(n_batch):
                xs,ys=mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:xs,y_:ys,keep_prob:1.0})
            learning_rate=sess.run(lr)
            acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
            print("After {} epochs, accuracy on test is {}, learning rate is {}".format(epoch,acc,learning_rate))



def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()