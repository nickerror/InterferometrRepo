import torch
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
from decimal import Decimal
import numpy as np
class CaptumVisualisation:
    def __init__(self, model_ft, modelOutputNumebr = 1) -> None:
        """Class shows the gradient of the most important parts of the photo most important for the model. Works only with hardtanh activation function.
        Args:
            model_ft (resnet): Model to predict value
            modelOutputNumebr (int, optional): number of neurons in the last layer
        """
        #ToDo: write method to display gradient on image.
        self.model_ft = model_ft
        self.ig = IntegratedGradients(model_ft) #captum
        self.modelOutputNumebr = modelOutputNumebr

    def showCaptumVisualisation(self, image, label, prediction, error, visualize = False, showOriginalImage = False):
        roundDigits = 4
        plotName = "epsilon = " + str(np.round_(label.cpu().detach().numpy(),roundDigits)) + ", prediction = " + str(np.round_(prediction.cpu().detach().numpy(),roundDigits)) + ", error = " + str(np.round_(error,roundDigits))
        if visualize == True:
            attributions = torch.empty((self.modelOutputNumebr,1,3,224,224), dtype=torch.float64)
            for attributeNr in range(self.modelOutputNumebr):
                attributions[attributeNr] = self.ig.attribute(inputs= image)#, target=attributeNr)
            averageAttributions = torch.sum(attributions,0)/self.modelOutputNumebr

            if showOriginalImage:
                fig, imgplot = plt.subplots(1,2)
                #plt.title(plotName,  loc='right')
                imgplot[0].imshow(image[0].cpu().numpy()[0], cmap='gray')
                imgplot[0].set_title("original image")
                imgplot[1].imshow(averageAttributions[0].cpu().numpy()[0], cmap='GnBu')
                imgplot[1].set_title("captum image")
                plt.axis('off')
                print("close image to continue ...")
                fig.suptitle(plotName)
                plt.show()
            else:
                imgplot = plt.imshow(averageAttributions[0].cpu().numpy()[0], cmap='GnBu')
                plt.axis('off')
                plt.title(plotName)
                print("close image to continue ...")
                plt.show()
            
            

    def showSingleNeuronCaptumVisualisation(self,image, neuronNo):
        attribution = self.ig.attribute(inputs=image, target=neuronNo)
        plt.imshow(attribution[0].cpu().numpy()[0], cmap='gray')