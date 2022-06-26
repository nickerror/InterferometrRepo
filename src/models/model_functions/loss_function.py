from typing import List
import torch

def custom_loss_function_debug(outputs, labels, totalLoss):
    print("NextOne")
    for i in range (len(outputs)):
        print("i: ", i, "label: ", float(labels[i]), "output:", float(outputs[i]), "diff= ", float(min( abs(abs(labels[i])-abs(outputs[i])) , abs(1-(abs(labels[i])-abs(outputs[i]))) )))

    print("totalLoss:", float(totalLoss))
    return totalLoss


def custom_loss_function(outputs, labels):
    totalLoss=0.0
    for i in range (len(outputs)):
        oneOutputLoss = torch_single_custom_loss_function(outputs[i], labels[i])
        totalLoss+=oneOutputLoss
    totalLoss/=len(outputs)
    return totalLoss

def single_custom_loss_function(outputs, labels):
    return min(abs(1-abs(float(labels-outputs))) , abs(float(labels-outputs)))

def torch_single_custom_loss_function(outputs, labels):
    return torch.min(torch.abs(1-torch.abs(labels-outputs)) , torch.abs(labels-outputs))

def total(xs: List[float]) -> float:
    result: float = 0.0
    for x in xs:
        result += x
    return result
