# encoding=utf-8
"""
保存所有变量的取值
"""
import tensorflow as tf
reader=tf.train.NewCheckpointReader("/tmp/model/dict/test.ckpt")
all_variables=reader.get_variable_to_shape_map()
print(all_variables)
for v_name in all_variables:
    print(v_name,all_variables[v_name])

print("Value for variable v1 is: ", reader.get_tensor("v1"))