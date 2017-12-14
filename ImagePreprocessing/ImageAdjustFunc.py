# encoding=utf-8
"""
TensorFlow提供了对jpeg和png格式图像的编码/解码
色相决定显示什么颜色
饱和度决定颜色的浓淡
亮度决定照射在颜色上的白光有多亮
图像标准化：将图像上的亮度均值变为０，方差变为１
"""
import tensorflow as tf
import matplotlib.pyplot as plt

#读取图像的原始数据
image_raw_data=tf.gfile.FastGFile("/tmp/data/dog.jpg",'rb').read()
with tf.Session() as sess:
    image_data=tf.image.decode_jpeg(image_raw_data)
    # plt.imshow(image_data.eval())
    # plt.show()
    encoded_img=tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile('/tmp/data/dog01.jpg','wb') as f:
        f.write(encoded_img.eval())
    image_data=tf.image.convert_image_dtype(image_data,dtype=tf.float32)
    # print(image_data.eval().shape)
    # resized_img1 = tf.image.resize_images(image_data, [300, 300], method=1)
    # resized_img2 = tf.image.resize_images(image_data, [300, 300], method=2)
    # resized_img3 = tf.image.resize_images(image_data, [300, 300], method=3)
    # print(resized_img1.eval().shape)
    # print(resized_img2.eval().shape)
    # print(resized_img3.eval().shape)
    #
    # # plt.imshow(image_data.eval())
    # # plt.show()
    # # plt.imshow(resized_img1.eval())
    # # plt.imshow(resized_img2.eval())
    # # plt.imshow(resized_img3.eval())
    # # plt.show()
    # croped=tf.image.resize_image_with_crop_or_pad(image_data,300,250)
    # padded=tf.image.resize_image_with_crop_or_pad(image_data,500,600)
    # # plt.imshow(padded.eval())
    # # plt.show()
    #
    # central_cropped=tf.image.central_crop(image_data,0.5)
    # # plt.imshow(central_cropped.eval())
    # # plt.show()
    #
    # #图像翻转
    # flipped_up_down=tf.image.random_flip_up_down(image_data)
    # # plt.imshow(flipped_up_down.eval())
    # # plt.show()
    # flipped_left_right=tf.image.flip_left_right(image_data)
    # # plt.imshow(flipped_left_right.eval())
    # # plt.show()
    # flipped_transposed=tf.image.transpose_image(image_data)
    # # plt.imshow(flipped_transposed.eval())
    # # plt.show()

    adjusted_bright_img=tf.image.adjust_brightness(image_data,0.3)
    # plt.imshow(adjusted_bright_img.eval())
    # plt.show()
    adjusted_random_bright_img=tf.image.random_brightness(image_data,0.5)
    # plt.imshow(adjusted_random_bright_img.eval())
    # plt.show()
    adjusted_contrast=tf.image.adjust_contrast(image_data,500)
    # plt.imshow(adjusted_contrast.eval())
    # plt.show()
    adjusted_hue=tf.image.adjust_hue(image_data,10.5)
    # plt.imshow(adjusted_hue.eval())
    # plt.show()
    adjusted_saturation=tf.image.adjust_saturation(image_data,-10)
    # plt.imshow(adjusted_saturation.eval())
    # plt.show()
    adjusted_standard=tf.image.per_image_standardization(image_data)
    plt.imshow(adjusted_standard.eval())
    plt.show()

    shape = tf.shape(image_data).eval()
    print(shape)
    h, w = shape[0], shape[1]


    fig = plt.figure()
    fig1 = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('orginal image')
    # ax.imshow(image_data)
    ax1 = fig1.add_subplot(311)
    ax1.set_title('original hist')
    ax1.hist(sess.run(tf.reshape(image_data, [h * w, -1])))
    ax1 = fig1.add_subplot(313)
    ax1.set_title('standardization hist')
    ax1.hist(sess.run(tf.reshape(adjusted_standard, [h * w, -1])))
    plt.ion()
    plt.show()
