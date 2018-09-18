# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_LeNet5

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径及文件名
MODEL_SAVE_PATH = "model_LeNet5/"
MODEL_NAME = "model_LeNet5.ckpt"

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference_LeNet5.IMAGE_SIZE, mnist_inference_LeNet5.IMAGE_SIZE,
                                    mnist_inference_LeNet5.NUM_CHANNELS], name= 'x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference_LeNet5.OUTPUT_NODE], name = 'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inference_LeNet5.inference(x, 1, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，学习率，滑动平均操作及训练过程
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference_LeNet5.IMAGE_SIZE,
                                          mnist_inference_LeNet5.IMAGE_SIZE, mnist_inference_LeNet5.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step= global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()