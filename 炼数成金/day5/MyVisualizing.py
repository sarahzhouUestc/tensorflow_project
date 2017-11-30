# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
num=3000
epochs=2000
sess=tf.Session()
embedding=tf.Variable(tf.stack(mnist.test.images[:num]),trainable=False,name="embedding")

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("sarah_mean",mean)
        tf.summary.scalar("sarah_stddev",tf.sqrt(tf.reduce_mean(tf.square(var-mean))))
        tf.summary.scalar("sarah_max",tf.reduce_max(var))
        tf.summary.scalar("sarah_min",tf.reduce_min(var))
        tf.summary.histogram("sarah_histogram",var)

#输入占位符
with tf.name_scope("sarah_input"):
    x=tf.placeholder(tf.float32,[None,784],name="x-input")
    y_=tf.placeholder(tf.float32,[None,10],name="y-input")
#网络定义
with tf.name_scope("layer"):
    with tf.name_scope("weights"):
        w=tf.Variable(tf.truncated_normal([784,10],stddev=0.1),dtype=tf.float32,name="w")
        variable_summaries(w)
    with tf.name_scope("biases"):
        b=tf.Variable(tf.constant(0.0,tf.float32,[10]),name="b")
        variable_summaries(b)
    wx=tf.matmul(x,w)+b
    prediction=tf.nn.softmax(wx)
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,axis=1),logits=prediction))
    tf.summary.scalar("myloss",loss)
with tf.name_scope("train"):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with tf.name_scope("accuracy"):
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y_,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("sarahacc",acc)

sess.run(tf.global_variables_initializer())

#产生metadata文件，里面存放正确label的信息
if tf.gfile.Exists("/tmp/data/projector/projector/metadata.tsv"):
    tf.gfile.DeleteRecursively("/tmp/data/projector/projector")
tf.gfile.MkDir("/tmp/data/projector/projector")
with open("/tmp/data/projector/projector/metadata.tsv","w") as f:
    labels=sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(num):
        f.write(str(labels[i])+"\n")

#在tensorboard中显示图片
with tf.name_scope("image_reshape"):
    image_reshaped=tf.reshape(x,[-1,28,28,1],name="image")
    tf.summary.image("image",image_reshaped,max_outputs=10)

merged=tf.summary.merge_all()

#产生event文件
projector_writer=tf.summary.FileWriter("/tmp/data/projector/projector",sess.graph)
config=projector.ProjectorConfig()
embed=config.embeddings.add()
embed.tensor_name=embedding.name
embed.metadata_path="/tmp/data/projector/projector/metadata.tsv"
embed.sprite.image_path="/tmp/data/mnist_10k_sprite.png"
embed.sprite.single_image_dim.extend([28,28])
#产生projector config文件
projector.visualize_embeddings(projector_writer,config)

saver=tf.train.Saver(max_to_keep=10)
for i in range(epochs):
    xs,ys=mnist.train.next_batch(100)
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata=tf.RunMetadata()
    summary,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,"sarahstep%03d"%i)
    projector_writer.add_summary(summary,i)
    if i%100 == 0:
        ac=sess.run(acc,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        saver.save(sess,"/tmp/data/projector/projector/test_model.ckpt",global_step=i)
        print("After iteration {}, the accuracy on test is {}".format(i,ac))

projector_writer.close()
sess.close()
