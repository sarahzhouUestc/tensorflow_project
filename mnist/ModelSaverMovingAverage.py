# encoding=utf-8
"""
演示使用saver的字典重命名来使用滑动平均值，替代 ema.average获取给定变量的滑动平均值
"""

# import tensorflow as tf
# # 以下是模型的从保存
# v=tf.Variable(0,dtype=tf.float32,name="v")
# for variable in tf.global_variables():
#     print(variable.name)
# ema=tf.train.ExponentialMovingAverage(0.99)
# ema_op=ema.apply(tf.global_variables())         #生成了影子变量 v/ExponentialMovingAverage:0
# for variable in tf.global_variables():
#     print(variable.name)
#
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     sess.run(tf.assign(v,100))
#     sess.run(ema_op)
#     print(sess.run([v,ema.average(v)]))
#     saver.save(sess,"/tmp/model/test/average/average.ckpt")

# 以下是模型的重命名加载
# import tensorflow as tf
# v=tf.Variable(0,dtype=tf.float32,name="v")
# saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
# with tf.Session() as sess:
#     saver.restore(sess,"/tmp/model/test/average/average.ckpt")
#     print(sess.run(v))

#使用ExponenialMovingAverage的variables_to_restore来生成tf.train.Saver类所需要的变量重命名字典
# import tensorflow as tf
# v=tf.Variable(0,dtype=tf.float32,name="v")
# ema=tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore(moving_avg_variables=[v]))
# print(ema.variables_to_restore())
# saver=tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     saver.restore(sess,"/tmp/model/test/average/average.ckpt")
#     print(sess.run(v))




