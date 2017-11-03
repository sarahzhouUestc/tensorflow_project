# encoding=utf-8
import tensorflow as tf

v1=tf.Variable(0,dtype=tf.float32)
step=tf.Variable(0,dtype=tf.int32,trainable=False)

#定义滑动平均的类，给定衰减率，和控制衰减率更新的num_updates即step
ema=tf.train.ExponentialMovingAverage(decay=0.99,num_updates=step)
#定义更新变量滑动平均的操作
moving_average_op=ema.apply(var_list=[v1])
with tf.Session() as sess:
    init_op=tf.global_variables_initializer();
    sess.run(init_op)    #初始化变量
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(v1,value=5))
    # tf.assign(v1,value=5)
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])

    # 更新step
    sess.run(tf.assign(step,value=10000))
    # 更新v1
    sess.run(tf.assign(v1,10))
    # 更新滑动平均类的操作
    sess.run(moving_average_op)
    # 打印影子和参数的更新结果
    print([sess.run(v1),sess.run(ema.average(v1))])


    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])
    sess.run(moving_average_op)
    print([sess.run(v1),sess.run(ema.average(v1))])

