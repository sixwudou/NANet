import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt #for the plots
from pathlib import Path
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#############################################
# GRAPH CREATION
#############################################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x  = tf.placeholder(tf.float32, [None, 784], name='x')

x_image = tf.reshape(x, [-1, 28, 28, 1]) # Image 28x28

W_conv1 = weight_variable([5, 5, 1, 32]) #3200 variables
b_conv1 = bias_variable([32]) #32 variables

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # Images 28x28
h_pool1 = max_pool_2x2(h_conv1) # Images 14x14

W_conv2 = weight_variable([5, 5, 32, 64]) #51.200 variables
b_conv2 = bias_variable([64]) #64 variables

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # Images 14x14
h_pool2 = max_pool_2x2(h_conv2) # Images 7x7

W_fc1 = weight_variable([7 * 7 * 64, 1024]) #3.211.264 variables
b_fc1 = bias_variable([1024]) #1024 variables

h_pool_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10]) #10.240 variables
b_fc2 = bias_variable([10]) #10 variables

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

W_fcA = weight_variable([28*28*32, 512])
b_fcA = bias_variable([512])
W_fcB = weight_variable([14*14*64, 512])
b_fcB = bias_variable([512])
W_fcC = weight_variable([1024, 512])
b_fcC = bias_variable([512])

h_conv1_flat = tf.reshape(h_conv1, [-1, 28*28*32])
h_conv2_flat = tf.reshape(h_conv2, [-1, 14*14*64])

h_fcA = tf.nn.relu(tf.matmul(h_conv1_flat, W_fcA) + b_fcA)
h_fcB = tf.nn.relu(tf.matmul(h_conv2_flat, W_fcB) + b_fcB)
h_fcC = tf.nn.relu(tf.matmul(h_fc1_drop, W_fcC) + b_fcC)

h_fcABC = tf.concat([h_fcA, h_fcB, h_fcC], 1)

W_fcX = weight_variable([512*3, 1024])
b_fcX = bias_variable([1024])

h_fcX = tf.nn.relu(tf.matmul(h_fcABC, W_fcX) + b_fcX)

W_fcout = weight_variable([1024, 784])
b_fcout = bias_variable([784])

h_fcout = tf.nn.sigmoid(tf.matmul(h_fcX, W_fcout) + b_fcout)

y_ = tf.placeholder(tf.float32, [None, 10])
#loss1 = 0 * tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
loss2 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=h_fcout))

loss = loss2

eta = 1e-1
train_step = tf.train.AdamOptimizer(eta).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

with tf.Session() as sess:

    # Import old variables:
    saver = tf.train.Saver()

    max_steps = 5
    saver.restore(sess, "modelsCNN/digit_recog_CNN_model%s" % max_steps)

    bb = sess.run(b_conv1)
    print(bb)
    print(eta)

    # Transform old variables into constants:
    b1 = tf.constant(sess.run(b_conv1))

    # Create new variable:
    newvar = bias_variable([1])

    # Keep training
    for ii in range(5):
        batch = mnist.train.next_batch(32)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    bc = sess.run(b1)
    bbb = sess.run(b_conv1)
    print(bc-bb)
    print(bc-bbb)

    loss_train = sess.run(loss2,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
    print(loss_train)


















#
