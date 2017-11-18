# encoding=utf-8
"""
使用graph_util的 convert_variables_to_constants 函数将 GraphDef 简化并仅仅持久化 GraphDef
"""
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
# result=v1+v2
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     #导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
#     graph_def=tf.get_default_graph().as_graph_def()
#     print(graph_def)
#     output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,output_node_names=['add'])
#     print("=======================")
#     print(output_graph_def)
#     with tf.gfile.GFile("/tmp/model/convertconstant2/combined_model.pb","wb") as f:
#         f.write(output_graph_def.SerializeToString())

import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename="/tmp/model/convertconstant2/combined_model.pb"
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #在保存的时候给的是计算节点的名称add, 在加载的时候给的是张量的名称，是 add:0，表示计算节点的第一个输出
    result=tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))