# encoding=utf-8
"""
前向传播的模块
"""
import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def get_weights(shape, regularizer):
    weights=tf.get_variable("weights",shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))    #debug
    return weights

def inference(input_tensor, regularizer):
    #debug
    with tf.variable_scope("layer1"):
        weights=get_weights([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable("biases",shape=[LAYER1_NODE],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope("layer2"):
        weights=get_weights([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",shape=[OUTPUT_NODE],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    return layer2

