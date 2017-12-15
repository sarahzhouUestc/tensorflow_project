# encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(img,color_ordering=0):
    if color_ordering==0:
        img=tf.image.random_brightness(img,max_delta=32./255.)
        img=tf.image.random_saturation(img,lower=0.5,upper=1.5)
        img=tf.image.random_hue(img,max_delta=0.2)
        img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
    elif color_ordering==1:
        img=tf.image.random_saturation(img,lower=0.5,upper=1.5)
        img=tf.image.random_brightness(img,max_delta=32./255.)
        img=tf.image.random_contrast(img,lower=0.5,upper=1.5)
        img=tf.image.random_hue(img,max_delta=0.2)
    return tf.clip_by_value(img,clip_value_min=0.0,clip_value_max=1.0)

def preprocess_for_train(img,height,width,bbox):
    #如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox==None:
        #一个框
        bbox=tf.constant([[[0.0,0.0,1.0,1.0]]],dtype=tf.float32,shape=[1,1,4])
    if img.dtype != tf.float32:
        #将像素值转换成了0-1.0之间的小数
        img=tf.image.convert_image_dtype(img,dtype=tf.float32)
    begin,size,random_bbox=tf.image.sample_distorted_bounding_box(tf.shape(img),bounding_boxes=bbox)
    img_sliced=tf.slice(img,begin,size)
    img=tf.expand_dims(img,axis=0)
    img_random=tf.image.draw_bounding_boxes(img,random_bbox)

    img_sliced=tf.image.resize_images(img_sliced,[height,width],method=np.random.randint(4))
    img_sliced=tf.image.random_flip_up_down(img_sliced)
    img_sliced=distort_color(img_sliced,1)
    return img_sliced,img_random

img_raw_data=tf.gfile.FastGFile("/tmp/data/dog.jpg",mode="rb").read()
with tf.Session() as sess:
    img_data=tf.image.decode_jpeg(img_raw_data)
    boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    img1,img2=preprocess_for_train(img_data,299,299,boxes)
    print(img_data.eval().shape)

    plt.imshow(img1.eval())
    plt.show()
    plt.imshow(img2.eval().reshape([313,500,3]))
    plt.show()
    # plt.imshow(img1.eval())
    # plt.show()
    