# encoding=utf-8
"""
tensorboard将graph可视化
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 参数概要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def train(mnist):
    batch_size=100
    n_batch=mnist.train.num_examples//batch_size

    #定义变量
    with tf.name_scope("input"):
        x=tf.placeholder(tf.float32,[None,784],name="x-input")
        y_=tf.placeholder(tf.float32,[None,10],name="y-input")
        keep_prob=tf.placeholder(tf.float32)
        lr=tf.Variable(0.001,dtype=tf.float32)

    #hidden_layer1
    with tf.name_scope("layer1"):
        weights1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1),dtype=tf.float32)
        biase1=tf.Variable(tf.constant(0.0,tf.float32,[500]))
        L1=tf.nn.tanh(tf.matmul(x,weights1)+biase1)
        L1_drop=tf.nn.dropout(L1,keep_prob)
        variable_summaries(weights1)
        variable_summaries(biase1)

    #hidden_layer2
    with tf.name_scope("layer2"):
        weights2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1),dtype=tf.float32)
        biase2=tf.Variable(tf.constant(0.0,tf.float32,[300]))
        L2=tf.nn.tanh(tf.matmul(L1_drop,weights2)+biase2)
        L2_drop=tf.nn.dropout(L2,keep_prob)
        variable_summaries(weights2)
        variable_summaries(biase2)

    #hidden_layer3
    with tf.name_scope("output"):
        weight3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1),dtype=tf.float32)
        biase3=tf.Variable(tf.constant(0.0,tf.float32,[10]))
        y=tf.matmul(L2_drop,weight3)+biase3
        variable_summaries(weight3)
        variable_summaries(biase3)

    #交叉熵损失函数
    with tf.name_scope("crossentropyloss"):
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        loss=tf.reduce_mean(cross_entropy,axis=0)
        tf.summary.scalar("loss",loss)

    #准确率
    with tf.name_scope("accuracy"):
        correct_prediction=tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),axis=0)
        tf.summary.scalar("acc",accuracy)

    with tf.name_scope("train"):
        train_step=tf.train.AdamOptimizer(lr).minimize(loss)
    init_op=tf.global_variables_initializer()

    merged=tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init_op)
        writer=tf.summary.FileWriter("/tmp/model/day5",graph=sess.graph)
        for epoch in range(20):
            sess.run(tf.assign(lr,0.001*(0.95**epoch)))
            for batch in range(n_batch):
                xs,ys=mnist.train.next_batch(batch_size)
                summary,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys,keep_prob:1.0})
                writer.add_summary(summary,epoch)
            learning_rate=sess.run(lr)
            acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
            print("After {} epochs, accuracy on test is {}, learning rate is {}".format(epoch,acc,learning_rate))



def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()