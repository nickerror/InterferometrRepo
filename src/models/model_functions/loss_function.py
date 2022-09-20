from typing import List
import torch

def custom_loss_function(outputs, labels):
    totalLoss=0.0
    for i in range (len(outputs)):
        oneOutputLoss = torch_single_custom_loss_function(outputs[i], labels[i])
        totalLoss+=oneOutputLoss
    totalLoss/=len(outputs)
    return totalLoss

def numpy_single_custom_loss_function(output, label):
    return torch_single_custom_loss_function(torch.tensor([output]), torch.tensor([label])).numpy()[0] #for linear activation
    #return torch_single_custom_loss_function(torch.tensor([output[0]]), torch.tensor([label])).numpy()[0] #for sigmoid


def torch_single_custom_loss_function(output, label):
    #return torch.min(torch.abs(1-torch.abs(label-output)) , torch.abs(label-output))
    return (1-torch.cos((torch.abs(output-label))*2*torch.pi))/2

def total(xs: List[float]) -> float:
    result: float = 0.0
    for x in xs:
        result += x
    return result
