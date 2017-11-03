# encoding=utf-8
import tensorflow as tf

a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')

result=tf.add(a,b,name='myadd')
print(result)

'''
c=tf.constant([1,2],name='c',dtype=tf.float32)
d=tf.constant([2.0,3.0],name='d')
result1=tf.add(c,d,name='myadd1')
print(result1)
print(result1.__getattribute__("shape"))

sess=tf.Session()
r=sess.run(result)
r1=sess.run(result1)
print(type(r))
print(dir(r1))
print(r.dtype)
sess.close()

#使用环境管理器来管理这个会话
with tf.Session() as sess:
    sess.run(result)
#不需要再调用close，因为在exit中会执行

sess=tf.Session()
sess.as_default()
with sess.as_default(): #指定默认的session，才能使用tensor的eval函数来计算
    print("=============session as default==============")
    print(result.eval())

print(result1.eval(session=sess))
print(type(result))

with tf.Session():
    print("============with default=============")
    print(result.eval())

'''
config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
sess=tf.Session(config=config)
print('================session with config proto================')
print(result.eval(session=sess))
sess.close()



