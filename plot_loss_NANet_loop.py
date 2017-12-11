# Plot loss results from digit_recog_CNN.py
import matplotlib.pyplot as plt
import numpy as np

crs = open("digit_recog_NANet_loop_update_RecF100.txt", "r")

step = []
loss_train = []
loss_test = []
train_acc = []
test_acc = []
for line in crs:
    step.append(line.split()[0])
    #print(step)
    loss_train.append(line.split()[1])
    loss_test.append(line.split()[2])
    train_acc.append(line.split()[3])
    test_acc.append(line.split()[4])

crs.close()

# Eliminate two heading rows
#In python2
step = map(float,step[2:])
loss_train = map(float,loss_train[2:])
loss_test = map(float,loss_test[2:])
train_acc = map(float,train_acc[2:])
test_acc = map(float,test_acc[2:])
#In python3
#step = list(map(float,step[2:]))
#loss_train = list(map(float,loss_train[2:]))
#loss_test = list(map(float,loss_test[2:]))
#train_acc = list(map(float,train_acc[2:]))
#test_acc = list(map(float,test_acc[2:]))

# 1-test
loss_test = [1-x for x in loss_test]
loss_train = [1-x for x in loss_train]


plt.figure(1)
plt.plot(step,loss_train,label="Training")
plt.plot(step,loss_test,label="Test")
plt.ylabel('1-loss2')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
#plt.yscale('log')
plt.grid(b=True,which='major')
plt.minorticks_on
plt.grid(b=True,which='minor')

plt.figure(2)
plt.plot(step,train_acc,label="Training")
plt.plot(step,test_acc,label="Test")
plt.ylabel('Accuracy')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
plt.yscale('log')
plt.grid(b=True,which='major')
plt.minorticks_on
plt.grid(b=True,which='minor')



plt.show()
