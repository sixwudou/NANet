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

from NANet import run_NANet

#############################################
# PARAMETERS
#############################################
# TRAINING NETWORK
# The CNN will only be trained on its own up to the transition step
max_steps = raw_input("Max steps CNN: ")
eta = raw_input("NANet learning rate: ")
#etaCNN = raw_input("CNN learning rate:")
trans_step = raw_input("Transition step: ")
RecF = raw_input("RecNet Factor: ")


# EVALUATION NETWORK
importRecNet = raw_input("Import pretrained RecNet? (Y/N) ")

#############################################
# RUN NETWORK
#############################################
run_NANet(max_steps, eta, trans_step, RecF, importRecNet)
