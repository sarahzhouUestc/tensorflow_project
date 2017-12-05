# encoding=utf-8
"""
使用卷积神经网络实现 mnist 图片分类
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR="/tmp/data"
EPOCHS=3000
BATCH_SIZE=100

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),tf.float32)

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,tf.float32,shape))

def variables_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)
        tf.summary.scalar("stddev",tf.sqrt(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar("max",tf.reduce_max(var))
        tf.summary.scalar("min",tf.reduce_min(var))
        tf.summary.histogram("histogram",var)

def train(mnist):
    with tf.name_scope("input"):
        x=tf.placeholder(tf.float32,[None,784],name="x-input")
        y_=tf.placeholder(tf.float32,[None,10],name="y-input")
        with tf.name_scope("x_reshaped"):
            x_reshaped=tf.reshape(x,[-1,28,28,1])

    with tf.name_scope("conv1"):
        #conv1
        w_conv1=weight_variable([5,5,1,32])
        variables_summaries(w_conv1)
        b_conv1=bias_variable([32])
        variables_summaries(b_conv1)
        h_conv1=tf.nn.relu(tf.nn.conv2d(x_reshaped,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)

    with tf.name_scope("pool1"):
        #pool1
        h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope("conv2"):
        #conv2
        w_conv2=weight_variable([5,5,32,64])
        variables_summaries(w_conv2)
        b_conv2=bias_variable([64])
        variables_summaries(b_conv2)
        h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)

    with tf.name_scope("pool2"):
        #pool2
        h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope("fc1"):
        #FC1,此时以上卷积层和池化层得到的特征平面是7x7x64
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
        w_fc1=weight_variable([7*7*64,1024])
        variables_summaries(w_fc1)
        b_fc1=bias_variable([1024])
        variables_summaries(b_fc1)
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
        with tf.name_scope("dropout"):
            #dropout层
            keep_prob=tf.placeholder(tf.float32)
            h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)

    with tf.name_scope("fc2"):
        #FC2,此时以上的结果是[-1,1024]
        w_fc2=weight_variable([1024,10])
        variables_summaries(w_fc2)
        b_fc2=bias_variable([10])
        variables_summaries(b_fc2)
        prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

    with tf.name_scope("loss"):
        #交叉熵
        cross_entroy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=prediction))
        tf.summary.scalar("crossentropy",cross_entroy)

    #优化器
    train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entroy)

    with tf.name_scope("accuracy"):
        #准确率
        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1),tf.argmax(prediction,axis=1)),tf.float32))
        tf.summary.scalar("acc",accuracy)

    merged=tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer=tf.summary.FileWriter("/tmp/data/day6/train",graph=sess.graph)
        test_writer=tf.summary.FileWriter("/tmp/data/day6/test",graph=sess.graph)
        for i in range(1001):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict={x:xs,y_:ys,keep_prob:0.7})
            train_summary=sess.run(merged,feed_dict={x:xs,y_:ys,keep_prob:1.0})
            train_writer.add_summary(train_summary,global_step=i)

            txs,tys=mnist.test.next_batch(BATCH_SIZE)
            test_summary=sess.run(merged,feed_dict={x:txs,y_:tys,keep_prob:1.0})
            test_writer.add_summary(test_summary,global_step=i)
            if i%100==0:
                train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images[:10000],y_:mnist.train.labels[:10000],keep_prob:1.0})
                test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
                print("After {} steps, the accuracy on train data is {}, the accuracy on test is{}".format(i,train_acc,test_acc))


def main(argv=None):
    mnist=input_data.read_data_sets(DATA_DIR, one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()