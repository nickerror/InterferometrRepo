#from model_functions.loss_function import custom_loss_function, single_custom_loss_function
import numpy as np


def custom_loss_function(outputs, labels):
    oneOutputLoss = min(abs(abs(labels[i])-abs(outputs[i])) , abs(1-(abs(labels[i])-abs(outputs[i]))))
    return oneOutputLoss 


def my_loss_function(outputs, labels):
    diff = float(min( abs(abs(float(labels[i])-abs(outputs[i]))) , abs(1-float((abs(labels[i])-abs(outputs[i]))))))
    return diff

def p_loss_function(outputs, labels):
    diff=min(abs(1-abs(float(labels[i]-outputs[i]))) , abs(float(labels[i]-outputs[i])))
    return diff

label = 0.903
outputs =np.full(1000,label) 


labels = np.zeros(1000)

results = np.zeros(1000)
results2 = np.zeros(1000)
results3 = np.zeros(1000)

labels[0] = 0.0
for i in range(999):
    labels[i+1] = labels[i] +0.001
    results[i] = custom_loss_function(outputs, labels)
    results2[i] = my_loss_function(outputs, labels)
    results3[i] = p_loss_function(outputs, labels)
    print(np.round(labels[i],3), ":              ", np.round(results[i],3), "      ;       " , np.round(results2[i],3), "      ;       " , np.round(results3[i],3))


