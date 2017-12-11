# Plot loss results from digit_recog_CNN.py
import matplotlib.pyplot as plt
import numpy as np

crs = open("NANet_clear.txt", "r")

step = []
loss1_train = []
loss1_test = []
test_acc = []
loss2_train = []
loss2_test = []
for line in crs:
    step.append(line.split()[0])
    #print(step)
    loss1_train.append(line.split()[1])
    loss1_test.append(line.split()[2])
    test_acc.append(line.split()[3])
    loss2_train.append(line.split()[4])
    loss2_test.append(line.split()[5])

crs.close()

# Eliminate heading row
#In python2
step = map(float,step[1:])
step_acc = map(float,step[4:])
loss1_train = map(float,loss1_train[1:])
loss1_test = map(float,loss1_test[1:])
test_acc = map(float,test_acc[5:])
loss2_train = map(float,loss2_train[1:])
loss2_test = map(float,loss2_test[1:])
#In python3
#step = list(map(float,step[2:]))
#loss_train = list(map(float,loss_train[2:]))
#loss_test = list(map(float,loss_test[2:]))
#train_acc = list(map(float,train_acc[2:]))
#test_acc = list(map(float,test_acc[2:]))

plt.figure(1)
plt.plot(step,loss1_train,label="Training")
plt.plot(step,loss1_test,label="Test")
plt.ylabel('loss1')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
plt.yscale('log')
plt.grid(b=True,which='major')
plt.minorticks_on
plt.grid(b=True,which='minor')

plt.figure(2)
plt.plot(step,loss2_train,label="Training")
plt.plot(step,loss2_test,label="Test")
plt.ylabel('loss2')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
#plt.yscale('log')
plt.grid(b=True,which='major')
plt.minorticks_on
plt.grid(b=True,which='minor')

plt.figure(3)
plt.plot(step_acc,test_acc,label="Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
#plt.yscale('log')
plt.grid(b=True,which='major')
plt.minorticks_on
plt.grid(b=True,which='minor')



plt.show()
