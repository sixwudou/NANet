##########################################################
# DIGIT RECOGNITION - NANET
##########################################################


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
# First layer: 3 fc layers of 512 units
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
# Loss CNN
loss1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Loss RecNet
loss2 = 0*tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=h_fcout))

loss = loss1+loss2

#############################################
# TRAINING
#############################################
# The argument of the gradient descent is the learning rate:
eta = 1e-4
train_step = tf.train.AdamOptimizer(eta).minimize(loss)

#############################################
# TEST
#############################################

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100

#############################################
# RUN MODEL
#############################################

max_steps = 2 #20000

with tf.Session() as sess:
  # Save or restore model
  saver = tf.train.Saver()
  my_file = Path("modelsCNN/digit_recog_CNN_model%s.meta" % max_steps)

  if my_file.is_file():
      # file exists
      saver.restore(sess, "modelsCNN/digit_recog_CNN_model%s" % max_steps)
      print("Model restored.")

      answer = input("Keep training restored model? (Y/N)")
      if answer in ['y', 'Y', 'yes', 'Yes', 'YES']:
            sum_steps_ = input("Number of extra training steps:")
            sum_steps = sess.run(tf.constant(eval(sum_steps_)))
            for step in range(sum_steps):
              batch = mnist.train.next_batch(32)

              if step % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                loss_train = sess.run(loss,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
                loss_test = sess.run(loss,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
                print('step %d, train_accuracy %f %%, test_accuracy %f %%,\n loss_train %f, loss_test %f' % (step+max_steps, train_accuracy,test_accuracy,loss_train,loss_test))
                file = open("digit_recog_CNN_results.txt","a")
                file.write("\n%f %f %f %f %f" %(step+max_steps,loss_train,loss_test,train_accuracy,test_accuracy))
                print("Results successfully recorded!")


              train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
              stats = np.random.poisson(1.2, 3)
              print("Level up! LVL %d: INT +%d, WIS +%d, DEX +%d, NEURONS +0" % (step+max_steps, stats[0], stats[1], stats[2]))

            steps_save = max_steps+sum_steps
            save_path = saver.save(sess, "modelsCNN/digit_recog_CNN_model%s" % steps_save)
            print("Model saved in file: %s" % save_path)
      else:
            print("Old model in use.")

  else:
      sess.run(tf.global_variables_initializer())

      for step in range(max_steps):
        batch = mnist.train.next_batch(32)

        if step % 1000 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0})
          test_accuracy = accuracy.eval(feed_dict={
              x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
          loss_train = sess.run(loss,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
          loss_test = sess.run(loss,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
          print('step %d, train_accuracy %f %%, test_accuracy %f %%,\n loss_train %f, loss_test %f' % (step, train_accuracy,test_accuracy,loss_train,loss_test))
          file = open("digit_recog_CNN_results.txt","a")
          file.write("\n%f %f %f %f %f" %(step,loss_train,loss_test,train_accuracy,test_accuracy))
          print("Results successfully recorded!")


        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        stats = np.random.poisson(1.2, 3)
        print("Level up! LVL %d: INT +%d, WIS +%d, DEX +%d, NEURONS +0" % (step, stats[0], stats[1], stats[2]))

      save_path = saver.save(sess, "modelsCNN/digit_recog_CNN_model%s" % max_steps)
      print("Model saved in file: %s" % save_path)



########################################
# COMPUTE TEST ACCURACY AND LOSS
########################################
  # test_accuracy = accuracy.eval(feed_dict={
  #       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  # loss_test = sess.run(loss,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
  # print("Test loss: %f" % loss_test)
  # print("Test accuracy: %f" % test_accuracy)


########################################
# CLASSIFY NEW SAMPLES
########################################
# # Get sample from the MNIST data set:
# x_new, onehot = mnist.train.next_batch(1)
# y_new = sess.run(tf.argmax(onehot,1))
#
# # Calculate solution:
# y_solution = sess.run(tf.argmax(y,1), feed_dict={x: x_new})
#
# #print(x_new)
# print("Actual label of the image (digit): %s" % y_new)
# print("Predicted label (digit): %s" % y_solution)
#
# ########################################
# # DISPLAY DATA
# ########################################
# def display_digit(x_new, y_new, y_solution):
#     #label = onehot.argmax(axis=0)
#     image = x_new.reshape([28,28])
#     if y_new == y_solution:
#         success = 'YES'
#     else:
#         success = 'NO'
#     plt.title('Actual label: %d Predicted label: %d Success: %s' % (y_new, y_solution, success))
#     plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#     plt.show()
#
# display_digit(x_new, y_new, y_solution)
