# encoding=utf-8
"""
完整的图像预处理流程
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"亮度、饱和度、色相、对比度的调整顺序对结果有影响"
def distort_color(image,color_ordering=0):
    if color_ordering==0:
        image=tf.image.random_brightness(image,max_delta=32./255.)
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image=tf.image.random_hue(image,max_delta=0.2)
        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
    if color_ordering==1:
        image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image=tf.image.random_brightness(image,max_delta=32./255.)
        image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image=tf.image.random_hue(image,max_delta=0.2)
    return tf.clip_by_value(image,0.0,1.0)

"只处理训练数据，对于测试数据，一般不需要使用随机变换的步骤"
def preprocess_for_train(sess,image,height,width,bbox):
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    #转换图像张量的类型
    if image.dtype != tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)
    #随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _=tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distored_image=tf.slice(image,bbox_begin,bbox_size)
    #将随机截取的图像调整为神经网络输入层的大小，大小调整的算法是随机选择的
    distored_image=tf.image.resize_images(distored_image,[height,width],method=np.random.randint(4))
    #随机左右翻转图像
    distored_image=tf.image.random_flip_left_right(distored_image)
    #使用一种随机的顺序调整图像色彩
    distored_image=distort_color(distored_image,np.random.randint(2))
    return distored_image

image_raw_data=tf.gfile.FastGFile("/tmp/data/dog.jpg","rb").read()
with tf.Session() as sess:
    img_data=tf.image.decode_jpeg(image_raw_data)
    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    #运行6次获得6种不同的图像，在图7-13展示了这些图像的效果
    for i in range(2):
        #将图像的尺寸调整为299x299
        result=preprocess_for_train(sess,img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()

