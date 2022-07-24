import torch
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt

class CaptumVisualisation:
    def __init__(self, model_ft, modelOutputNumebr = 512) -> None:
        """Class shows the gradient of the most important parts of the photo most important for the model. Works only with hardtanh activation function.
        Args:
            model_ft (resnet): Model to predict value
            modelOutputNumebr (int, optional): number of neurons in the last layer
        """
        #ToDo: write method to display gradient on image.
        self.model_ft = model_ft
        self.ig = IntegratedGradients(model_ft) #captum
        self.modelOutputNumebr = modelOutputNumebr

    def showCaptumVisualisation(self, image, visualize = False):
        if visualize == True:
            attributions = torch.empty((self.modelOutputNumebr,1,3,224,224), dtype=torch.float64)
            for attributeNr in range(self.modelOutputNumebr):
                attributions[attributeNr] = self.ig.attribute(inputs= image, target=attributeNr)
            averageAttributions = torch.sum(attributions,0)/self.modelOutputNumebr
            
            
            imgplot = plt.imshow(averageAttributions[0].cpu().numpy()[0], cmap='gray')
            print("close image to continue ...")
            plt.show()
            

    def showSingleNeuronCaptumVisualisation(self,image, neuronNo):
        attribution = self.ig.attribute(inputs=image, target=neuronNo)
        plt.imshow(attribution[0].cpu().numpy()[0], cmap='gray')