# encoding=utf-8
"""
定义了前向传播的过程以及神经网络中的参数
"""
import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights

#定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope("layer2"):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        #没有用激活函数激活
        layer2=tf.matmul(layer1,weights)+biases
    return layer2