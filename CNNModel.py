#网络模型构建

import tensorflow as tf

def weight_variable(shape, n):
    # tf.truncated_normal(shape, mean, stddev)函数产生正态分布。
    # shape表示生成张量的维度，mean是均值
    # stddev是标准差,，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial
 
def bias_variable(shape):
    # 创建一个数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial
 
def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    # 卷积层后输出图像大小为：（W+2P-f）/stride+1并向下取整
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x, name):
    # 池化层采用kernel大小为2*2，步数也为2，SAME：周围补0，取最大值。
    # x 是 CNN 第一步卷积的输出量，其shape必须为[batch, height, weight, channels];
    # ksize 是池化窗口的大小， shape为[batch, height, weight, channels]
    # stride 步长，一般是[1，stride， stride，1]
    # 池化层输出图像的大小为(W-f)/stride+1，向上取整
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# 搭建网络
def deep_CNN(images, batch_size, n_classes):
    # 第一层卷积
    # 64个3x3的卷积核，padding=’SAME’，表示padding卷积的图像与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        #卷积核为3*3
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')  # 得到128*128*64
    # 第一层池化
    # 2x2最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')   # 得到64*64*64
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
 
    # 第二层卷积
    # 32个3x3卷积核，padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')  # 得到64*64*32
    # 第二层池化
    # 2x2最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')  # 得到32*32*32
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
 
    # 第三层卷积
    # 16个3x3的卷积核，padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3)+b_conv3, name='conv3')  # 得到32*32*16
    # 第三层池化
    # 2x2最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3, 'pooling3')  # 得到16*16*16
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
 
    # 全连接层
    # 将pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)
 
    # 全连接层
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)

    # dropout层
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)
 
    # Softmax回归层
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
    return softmax_linear
    # 最后返回softmax层的输出

# loss计算
# 传入参数：logits，网络计算输出值。
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.name_scope('loss') :
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss', loss)
    return loss
 
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
 
# 准确率计算
# 输入参数：logits，网络计算值。
# 返回参数：accuracy，平均准确率。
def evaluation(logits, labels):
    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy
