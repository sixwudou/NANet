The RecNet may be used as a regularization method for the CNN. This folder contains the codes used to evaluate whether that would be possible or not.

main.py: execute this file to run the simulation. It will train a CNN (loss1) on its own (loss = loss1) up to a transition step, when it will keep training it but will also start to train a RecNet (loss2) simultaneously (loss = loss1 + RecF * loss2).
It will ask for some parameters in the following order: maximum number of steps, learning rate (it will be the same before and after the transition step), transition step, reconstruction factor (RecF) and import pretrained RecNet or not. 
If we don't import a RecNet, its variables will be initialized at the transition step. If we import a trained RecNet, the code will ask for the number of steps it was trained and the learning rate. To train a RecNet for this purpose, run NANet/digit_recog_NANet.py.
The results (training loss1, test loss1, test accuracy, training loss2 and test loss2 for each step) are written in a text file (NANet_clear.txt if we don't import a RecNet, NANet_imported.txt if we do). Each time we run main.py, a header will be placed before all the data containing the inputs we gave to the program.

NANet.py: code called from main.py containing all the functions required.

NANet_clear.txt and NANet_imported.txt: data files containing step, training loss1, test loss1, test accuracy, training loss2 and test loss2. Each data set is headed by a line containing the input data.

Plot_NANet_clear.py and Plot_NANet_imported.py: plot the results from the text files. It is important that only a single dataset has to be present in the text file (heading line and data lines).

Plots_X_Y_Z folders: they contain the plots of the results contained in the text files. The name of each folder is Plots_LearningRate_TransitionStep_RecF. Plots = no RecNet imported. PlotsImported = RecNet imported (I used a NANet trained for 40k steps with a learning rate of 1e-3). PlotsImportedMap2 = RecNet imported but only the NAs map #2 is used. PlotsNoRecNet = the transition step is never achieved (same as setting RecF=0).
