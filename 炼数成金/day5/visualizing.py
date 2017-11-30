# encoding=utf-8
"""
tensorboard可视化,embedding
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
epochs=3000
image_num=2000
DIR="/tmp/data/"
sess=tf.Session()
embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name="embedding")

#summaries
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean=tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev",stddev)
        tf.summary.scalar("max",tf.reduce_max(var)) #debug
        tf.summary.scalar("min",tf.reduce_min(var))
        tf.summary.histogram("histogram",var)
with tf.name_scope("input"):
    x=tf.placeholder(tf.float32,[None,784],name="x-input")
    y_=tf.placeholder(tf.float32,[None,10],name="y-input")

#显示图片
with tf.name_scope("input_reshape"):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image("input",image_shaped_input, 10)

with tf.name_scope("layer"):
    #创建一个简单神经网络
    with tf.name_scope("weights"):
        W=tf.Variable(tf.zeros([784,10]),name="W")
        variable_summaries(W)
    with tf.name_scope("biases"):
        b=tf.Variable(tf.zeros([10]),name="b")
        variable_summaries(b)
    with tf.name_scope("wx_plus_b"):
        wx_plus_b=tf.matmul(x,W)+b
    with tf.name_scope("softmax"):
        prediction=tf.nn.softmax(wx_plus_b)

with tf.name_scope("loss"):
    #交叉熵代价函数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=prediction))
    tf.summary.scalar("loss",loss)
with tf.name_scope("train"):
    #使用梯度下降法
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(prediction,1))
    with tf.name_scope("accuracy"):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar("accuracy",accuracy)

#产生metadata文件
if tf.gfile.Exists(DIR+"projector/projector/metadata.tsv"):
    tf.gfile.DeleteRecursively(DIR+"projector/projector/")
tf.gfile.MkDir(DIR+"projector/projector/")

#产生metadata.tsv文件，里面存放 label信息
with open(DIR+"projector/projector/metadata.tsv","w") as f:
    labels=sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):
        f.write(str(labels[i])+"\n")

#合并所有的summary
merged=tf.summary.merge_all()

#产生event文件
projector_writer=tf.summary.FileWriter(DIR+"projector/projector",sess.graph)
saver=tf.train.Saver(max_to_keep=10)
config=projector.ProjectorConfig()
embed=config.embeddings.add()
embed.tensor_name=embedding.name
embed.metadata_path=DIR+'projector/projector/metadata.tsv'
embed.sprite.image_path=DIR+"projector/data/mnist_10k_sprite.png"
embed.sprite.single_image_dim.extend([28,28])
#产生projector_config文件，里面存放的是config的protocol buffer的字符串信息
projector.visualize_embeddings(projector_writer,config)

for i in range(epochs):
    #每个批次100个样本
    xs,ys=mnist.train.next_batch(100)
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata=tf.RunMetadata()
    summary,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,"step%03d"%i)
    projector_writer.add_summary(summary,i)

    if i%100==0:
        ac=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        print("Iter "+str(i)+", Testing Accuracy= "+str(ac))
        #产生model相关的４个文件
        saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=i)

projector_writer.close()
sess.close()