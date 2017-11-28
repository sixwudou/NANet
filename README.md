# NANet
NeuroActivations Network
A Reconstruction Network (RecNet) is used to study the neuroactivations of a CNN. There are two ways of studying the network:
  -NANet: both NNs are trained together, the loss of the CNN and the RecNet are added to give the total loss.
  -CNN+RecNet: first train CNN, then RecNet*.

digit_recog_CNN.py: creates and trains the CNN.
digit_recog_RecNet.py: creates and trains the RecNet using an already trained CNN.
digit_recog_NANet.py: creates and trains NANet.
digit_recog_NANet_loop.py: evaluates loss2 of an already trained NANet using as input the NAs of a CNN which is being trained in the loop.
NANet_study_NAs_select.py: evaluate the results of NANet. It is possible to manually activate some NAs inside the code.
RecNet_study.py: evaluate the results of RecNet.
plot_loss_NANet_loop.py: plots the results from digit_recog_NANet_loop.py.

*CNN+RecNet combination is dangerous at the moment, since it creates too big graphs that TF cannot save (>2Gb)
