# encoding=utf-8
import tensorflow as tf
weights=tf.Variable(tf.random_normal([2,3],stddev=2))


'''
values=tf.random_normal([2,3])
with tf.Session():
    print(values.eval())

print(values)
# weights=tf.Variable()

zerosdemo=tf.zeros([2,3],dtype=tf.int32,name='zerosdemo')
onesdemo=tf.ones([2,3],dtype=tf.float32,name='onesdemo')
filldemo=tf.fill([3,4],value=5,name='filldemo')
consdemo=tf.constant([2,2],dtype=tf.float32,name='consdemo')

sess=tf.Session()
with sess.as_default():
    print(zerosdemo.eval())
    print(onesdemo.eval())

print(filldemo.eval(session=sess))
print(sess.run(consdemo))
'''
weights=tf.Variable(tf.random_normal([2,3],stddev=2))
a=tf.Variable(tf.zeros([3]))
b=tf.Variable(tf.ones([2,3]))
w2=tf.Variable(weights.initialized_value())
w3=tf.Variable(weights.initialized_value()*2)

print(a)
print(a.value().op)
print(weights.initialized_value())
print(b.value())
print(w3.value())
print(w2.value())



# with tf.Session():
    # print(a.eval())
    # print(b.eval())



