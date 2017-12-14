# encoding=utf-8
import tensorflow as tf
import matplotlib.image as image
import matplotlib.pyplot as plt

img=image.imread("/tmp/data/dog.jpg")
img_data=tf.image.resize_images(img,[180,267],method=1)
img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)

boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
begin,size,bbox_for_draw=tf.image.sample_distorted_bounding_box(tf.shape(img_data),bounding_boxes=boxes)
batched=tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
img_with_box=tf.image.draw_bounding_boxes(batched,bbox_for_draw)
distorted_img=tf.slice(img_data,begin,size)

with tf.Session() as sess:
    plt.imshow(distorted_img.eval())
    plt.show()
    plt.imshow(img_with_box.eval().reshape([180,267,3]))
    plt.show()


# print(img_data.shape)
# img_batched=tf.expand_dims(img_data,axis=0)
# print(img_batched.shape)
#
# boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
# result=tf.image.draw_bounding_boxes(img_batched,boxes)
#
#
# with tf.Session() as sess:
#     plt.imshow(result.eval().reshape([180,267,3]))
#     plt.show()
