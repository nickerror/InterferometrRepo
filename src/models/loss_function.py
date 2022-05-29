import torch

def customLossFunctionDebug(outputs, labels, totalLoss):
    print("NextOne")
    for i in range (len(outputs)):
        print("i: ", i, "label: ", float(labels[i]), "output:", float(outputs[i]), "diff= ", float(min( abs(abs(labels[i])-abs(outputs[i])) , abs(1-(abs(labels[i])-abs(outputs[i]))) )))

    print("totalLoss:", float(totalLoss))
    return totalLoss


def customLossFunction(outputs, labels):
    totalLoss=0.0
    for i in range (len(outputs)):
        #oneOutputLoss= abs(abs(labels[i])-(outputs[i]))
        #oneOutputLoss=min( abs(abs(labels[i])-abs(outputs[i])) , abs(1-(abs(labels[i])-abs(outputs[i]))))
        oneOutputLoss = torch.min( torch.abs(torch.abs(labels[i])-torch.abs(outputs[i])) , torch.abs(1-(torch.abs(labels[i])-torch.abs(outputs[i]))))
        totalLoss+=oneOutputLoss
    totalLoss/=len(outputs)
    #customLossFunctionDebug(outputs=outputs, labels=labels, totalLoss=totalLoss)
    return totalLoss

def singleCustomLossFunction(outputs, labels):
    return torch.min( torch.abs(torch.abs(labels)-torch.abs(outputs)) , torch.abs(1-(torch.abs(labels)-torch.abs(outputs))))