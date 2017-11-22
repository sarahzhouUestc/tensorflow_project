# encoding=utf-8
"""
定义训练过程
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

#配置神经网络的参数
BATCH_SIZE=100
#learning_rate=learning_rate_base*learning_rate_decay^(global_steps/decay_steps)
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEP=30000
#shadow_variable=moving_average_decay*shadow_variable+(1-moving_average_decay)*variable
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="/tmp/model/mnist"
MODEL_NAME="mnist.ckpt"

def train(mnist):
    x=tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_=tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,axis=1),logits=y)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op("train")

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEP):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step], feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("After %d training step(s), loss on training batch is %f"%(step,loss_value))
                #在保存的模型名称末尾加上训练的轮数
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()

