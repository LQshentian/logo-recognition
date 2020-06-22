#数据预处理

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *
 
adidas = []
label_adidas = []
nike = []
label_nike = []
puma = []
label_puma = []
supreme = []
label_supreme = []


def get_file(file_dir):   # 获取路径下所有的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中
    for file in os.listdir(file_dir + '/adidas'):   #用于返回指定的文件夹包含的文件或文件夹的名字的列表    
        adidas.append(file_dir + '/adidas' + '/' + file)
        label_adidas.append(0)
    for file in os.listdir(file_dir + '/nike'):
        nike.append(file_dir + '/nike' + '/' + file)
        label_nike.append(1)
    for file in os.listdir(file_dir + '/puma'):
        puma.append(file_dir + '/puma' + '/' + file)
        label_puma.append(2)
    for file in os.listdir(file_dir + '/supreme'):
        supreme.append(file_dir + '/supreme' + '/' + file)
        label_supreme.append(3)
      
    # 检测是否正确提取
    print("There are %d adidas\nThere are %d nike\nThere are %d puma\n" %(len(adidas), len(nike), len(puma)),end="")
    print("There are %d supreme\n" %(len(supreme)))
 
    # 对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 水平方向上（按列顺序）合并数组
    image_list = np.hstack((adidas, nike, puma, supreme))
    label_list = np.hstack((label_adidas, label_nike, label_puma, label_supreme))
    # 转置、随机打乱
    temp = np.array([image_list, label_list])   # 转换成二维矩阵
    temp = temp.transpose()     # 转置
    np.random.shuffle(temp)     #随机打乱
 
    # 将所有的img和lab转换成list  list:将元组转换为列表
    all_image_list = list(temp[:, 0])    # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])    # 取出第1列数据，即图片标签
    label_list = [int(i) for i in label_list]   # 转换成int数据类型
 
    return image_list, label_list  #返回两个list
 

# 为了方便网络的训练，输入数据进行batch处理
# image_W, image_H, ：图像高度和宽度
# batch_size：每个batch存放的图片
# capacity：队列
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)   # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)    #tf.cast强制类型转换  image->string   label->int32
    # tf.train.slice_input_producer为tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label]) #次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # tf.read_file()从队列中读取图像
 
    #图像解码
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 数据预处理
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
     # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)
 
   # 生成batch
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
 
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.uint8)  # 显示彩色图像
    # image_batch = tf.cast(image_batch, tf.float32)    # 显示灰度图
    return image_batch, label_batch
     # 获取两个batch，两个batch即为传入神经网络的数据

def PreWork():
    # 对预处理的数据进行可视化，查看预处理的效果
    IMG_W =128
    IMG_H = 128
    BATCH_SIZE = 6
    CAPACITY = 64
    train_dir = 'picture'
    image_list, label_list = get_file(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(label_batch.shape)
    lists = ('adidas', 'nike', 'puma', 'supreme')
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                # 提取出两个batch的图片并可视化
                img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label

                for j in np.arange(BATCH_SIZE):
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    title = lists[int(label[j])]
                    plt.title(title)
                    plt.show()
                i += 1

        finally:
            coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    PreWork()