# encoding=utf-8
import tensorflow as tf 
# import pudb
# pudb.set_trace()

#<class 'tensorflow.python.framework.ops.Graph'>
g1=tf.Graph()
print(g1.as_default())
with g1.as_default():
	v=tf.get_variable("v",initializer=tf.zeros_initializer()(shape=[1]))


g2=tf.Graph()
with g2.as_default():
	v=tf.get_variable("v",initializer=tf.ones_initializer()(shape=[1]))


with tf.Session(graph=g1) as sess:
	with tf.variable_scope("",reuse=True):
		pass
	tf.global_variables_initializer().run()
	print(sess.run(tf.get_variable("v")))


with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))

