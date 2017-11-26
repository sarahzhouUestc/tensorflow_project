# encoding=utf-8
"""测试模型"""
import tensorflow as tf
import mnist_inference
import mnist_train
from tensorflow.examples.tutorials.mnist import input_data
import time
EVAL_INTERVAL_SEC=10

def evaluate(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],"x-input")
    y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],"y-input")
    validate_feed={x:mnist.validation.images, y_:mnist.validation.labels}
    y=mnist_inference.inference(x,None)
    # 正确率
    correction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
    # 加载模型
    ema=tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    saver=tf.train.Saver(ema.variables_to_restore())
    while True:
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                print("After %s step(s), the accuracy on validation is %f"%(global_step,accuracy_score))
            else:
                print("No checkpoint found")
        time.sleep(EVAL_INTERVAL_SEC)


def main(argv=None):
    mnist=input_data.read_data_sets(mnist_train.DATA_PATH,one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()