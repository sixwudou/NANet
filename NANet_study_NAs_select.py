##########################################################
# DIGIT RECOGNITION - NANET STUDY NeuroActivations selection
##########################################################
# Plots the original and reconstructed digit, and also
#the NAs that THE USER SPECIFIED

#############################################
# IMPORT AND READ DATA
#############################################
import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt #for the plots
from pathlib import Path
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#############################################
# PREVIOUS DEFINITIONS
#############################################
# Initialization of parameters in a positive value to avoid "dead" neurons
#(We are using ReLU)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# CONVOLUTION AND POOLING:
# Our convolutions uses a stride of one and are zero padded so that the output
#is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# DISPLAY DATA
def display_digit(x_new, y_new, y_solution, reconstr):
    #label = onehot.argmax(axis=0)
    image_original = x_new.reshape([28,28])
    image_rec = reconstr.reshape([28,28])
    image = np.concatenate([image_original, image_rec], 1)
    #print(image_original)
    #print(image_rec)
    if y_new == y_solution:
        success = 'YES'
    else:
        success = 'NO'
    plt.title('Actual label: %d Predicted label: %d Success: %s' % (y_new, y_solution, success))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

def display_NAs(h):
    N0 = tf.cast(h.shape[3],tf.float32)
    N = sess.run(tf.to_int32(tf.ceil(tf.sqrt(N0))))
    print('Total number of slices order (subplots grid): %dx%d' % (N,N))
    f, axarr = plt.subplots(N,N)

    p = 0

    if N == 1:
        slc = sess.run(tf.reduce_mean(h[:,:,:,p],axis=0))
        plt.imshow(slc, cmap=plt.get_cmap('gray_r'))

    else:
        for i in range(N):
            for j in range(N):
                # Empty plot white when out of slices to display
                if p >= h.shape[3]:
                    slc = np.ones((h.shape[1],h.shape[3],3))
                else:
                    if h.shape[0] == 8:
                        slc = h[:,:,:,p]
                        axarr[i,j].imshow(slc)
                    else:
                        slc = sess.run(tf.reduce_mean(h[:,:,:,p],axis=0))
                        #print(slc.shape)
                        axarr[i,j].imshow(slc,cmap='Reds')
                axarr[i,j].axis('off')
                p+=1

#############################################
# CNN
#############################################
# Input layer:
x  = tf.placeholder(tf.float32, [None, 784], name='x')

# To apply the layer, we first reshape x to a 4d tensor, with the second
#and third dimensions corresponding to image width and height, and the final
#dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1]) # Image 28x28

# The convolution will compute 32 features for each 5x5 patch.
#Its weight tensor will have a shape of [5, 5, 1, 32]. The first two
#dimensions are the patch size, the next is the number of input channels,
#and the last is the number of output channels. We will also have a bias
#vector with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 32]) #3200 variables
b_conv1 = bias_variable([32]) #32 variables

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # Images 28x28
h_pool1 = max_pool_2x2(h_conv1) # Images 14x14

# Second layer (64 new features)
W_conv2 = weight_variable([5, 5, 32, 64]) #51.200 variables
b_conv2 = bias_variable([64]) #64 variables

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # Images 14x14
h_pool2 = max_pool_2x2(h_conv2) # Images 7x7

# Fully connected final layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) #3.211.264 variables
b_fc1 = bias_variable([1024]) #1024 variables

h_pool_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Dropout to reduce overfitting (With probability keep_prob, outputs the
#input element scaled up by 1 / keep_prob, otherwise outputs 0. The scaling
#is so that the expected sum is unchanged):
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer:
W_fc2 = weight_variable([1024, 10]) #10.240 variables
b_fc2 = bias_variable([10]) #10 variables

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2






#############################################
# RecNet
#############################################
# Inputs:
hconv1 = tf.placeholder(tf.float32, [None, 28,28,32], name='hconv1')
hconv2 = tf.placeholder(tf.float32, [None, 14,14,64], name='hconv2')
hfc1 = tf.placeholder(tf.float32, [None, 1024], name='hfc1')

# First layer: 3 fc layers of 512 units
W_fcA = weight_variable([28*28*32, 512])
b_fcA = bias_variable([512])
W_fcB = weight_variable([14*14*64, 512])
b_fcB = bias_variable([512])
W_fcC = weight_variable([1024, 512])
b_fcC = bias_variable([512])

h_conv1_flat = tf.reshape(hconv1, [-1, 28*28*32])
h_conv2_flat = tf.reshape(hconv2, [-1, 14*14*64])

h_fcA = tf.nn.relu(tf.matmul(h_conv1_flat, W_fcA) + b_fcA)
h_fcB = tf.nn.relu(tf.matmul(h_conv2_flat, W_fcB) + b_fcB)
h_fcC = tf.nn.relu(tf.matmul(hfc1, W_fcC) + b_fcC)

# Second layer: fc layer of 1024 units
h_fcABC = tf.concat([h_fcA, h_fcB, h_fcC], 1)

W_fcX = weight_variable([512*3, 1024])
b_fcX = bias_variable([1024])

h_fcX = tf.nn.relu(tf.matmul(h_fcABC, W_fcX) + b_fcX)

# Fully connected final layer
W_fcout = weight_variable([1024, 784])
b_fcout = bias_variable([784])

h_fcout = tf.nn.sigmoid(tf.matmul(h_fcX, W_fcout) + b_fcout)


#############################################
# COST FUNCTION (CROSS-ENTROPY)
#############################################
y_ = tf.placeholder(tf.float32, [None, 10])
# Loss 1 (cross-entropy)
loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Loss 2 (cross-entropy)
loss2 = tf.reduce_mean(
    tf.losses.mean_squared_error(labels=x, predictions=h_fcout))

loss = loss1 + 0.01*loss2

#############################################
# TRAINING
#############################################
# The argument of the gradient descent is the learning rate:
eta = 1e-3
train_step = tf.train.AdamOptimizer(eta).minimize(loss)

#############################################
# TEST
#############################################

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

#############################################
# RUN MODEL
#############################################

max_steps = 40000 # 5

with tf.Session() as sess:
    # Save or restore model
    saver = tf.train.Saver()
    my_file = Path("modelsNaNet/NANet_model_%s_%s.meta" % (max_steps,eta))

    if my_file.is_file():
      # file exists
      saver.restore(sess, "modelsNaNet/NANet_model_%s_%s" % (max_steps,eta))
      print("Model restored.")

      ########################################
      # GET A PARTICULAR DIGIT
      ########################################
      digit_ = raw_input("Number to study (0-9):")
      digit = sess.run(tf.constant(eval(digit_)))

      counter = 0
      while counter == 0:
          # Get sample from the MNIST data set:
          x_new, onehot = mnist.train.next_batch(1)
          y_new = sess.run(tf.argmax(onehot,1))
          #print(digit)
          #print(y_new)
          if y_new == digit:

              ########################################
              # GET A PARTICULAR DIGIT
              ########################################
              xx_select = tf.reshape(x_new, [-1, 28, 28, 1])
              hconv1_select = sess.run(tf.nn.relu(conv2d(xx_select, W_conv1) + b_conv1))
              hpool1_select = sess.run(max_pool_2x2(hconv1_select))
              hconv2_select = sess.run(tf.nn.relu(conv2d(hpool1_select, W_conv2) + b_conv2))
              hpool2_select = sess.run(max_pool_2x2(hconv2_select))
              hfc1_select = sess.run(tf.nn.relu(tf.matmul(tf.reshape(hpool2_select, [-1, 7*7*64]), W_fc1) + b_fc1))

              hconv1_select = hconv1_select
              hconv2_select = hconv2_select*0
              hfc1_select = hfc1_select*0 #sess.run(tf.square(hfc1_select))*0

              # Calculate solution:
              y_solution = sess.run(tf.argmax(y_conv,1), feed_dict={x: x_new, keep_prob: 1})
              # Reconstruction of image:
              reconstr = sess.run(h_fcout, feed_dict={x: x_new, keep_prob: 1, hconv1: hconv1_select, hconv2: hconv2_select, hfc1: hfc1_select})

              # Display original and reconstructed digit
              display_digit(x_new, y_new, y_solution, reconstr)

              # Display NeuroActivations
              hc1 = hconv1_select
              #print(hc1.shape)
              display_NAs(hc1)

              hc2 = hconv2_select
              display_NAs(hc2)

              hfc1 = hfc1_select
              hfc1 = tf.reshape(hfc1, [1,8,128,1])
              #print(hfc1.shape)
              #print(sess.run(tf.reduce_max(hfc1)))
              display_NAs(hfc1)

              # Plot everything
              plt.show()


              counter = 1

    else:
      print("Cannot find model.")
