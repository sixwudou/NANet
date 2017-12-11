# NANet
NeuroActivations Network: a Reconstruction Network (RecNet) is used to study the neuroactivations of a CNN. There are two ways of studying the network:
  
  -NANet: both NNs are trained together, the loss of the CNN (loss1) and the RecNet (loss2) are added to give the total loss (loss = loss1 + RecF * loss2, being RecF a parameter).
  
  -CNN+RecNet: first train CNN, then RecNet*.


digit_recog_NANet.py: creates, trains and saves NANet.

digit_recog_CNN.py: creates, trains and saves the CNN.

digit_recog_RecNet.py: creates, trains and saves the RecNet using an already trained CNN.

digit_recog_NANet_loop.py: evaluates loss2 of an already trained NANet using as input the NAs of a CNN which is being trained in the loop (OLD CODE, not recommended to use).

plot_loss_NANet_loop.py: plots the results from digit_recog_NANet_loop.py.

NANet_study_NAs_select.py: display the desired NAs (it is possible to manually deactivate NAs inside the code) that will be given as input for the RecNet. It also shows the reconstructed digit to evaluate if the NAs used are enough to get a good reconstruction. It requires an existent NANet model.

RecNet_study.py: plot the reconstructed digit.



*CNN+RecNet combination is dangerous at the moment, since it creates too big graphs that TF cannot save (>2Gb). The codes can run to get loss results but the networks cannot be saved.
