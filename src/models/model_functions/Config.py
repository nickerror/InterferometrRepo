from torchvision import transforms
import torch
#from PathManagement import PathManagement
class Config:
    def __init__(self, pathManagement, forTest = False):
        #Variables to edit
        self.epochs = 20      #number of epochs
        self.num_classes = 1  #num classes in dataset
        #todo zwiekszyc batch_size -> 16 -> 32
        self.batch_size = 4   #batch size used for training (e.g. bath_size photos in one process)
        #todo sprawdzic Adama
        self.learning_rate = 0.001 #for SGD = 0.01, for Adam = 10^-4 -- 10^-3
        self.train_size=0.8
        self.dataset = "InterferometerPhoto"
        #self.architecture = "CNN"
        self.pin_memory = True
        self.momentum = 0.9 #do Adama
        self.step_size = 7
        self.gamma = 0.1
        self.num_workers = 0
        self.model_name_to_save = "1_generated_unnoised.pth"
        self.model_name_to_read = "1_generated_unnoised.pth"
        self.data_place = "local" #="cloud"
        self.data_transforms = transforms.Compose([
                        transforms.CenterCrop(448),
                        transforms.Resize(224), #first way is crop and later resize. Second way is CenterCrop right away.
                        #transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.491, 0.491, 0.491],
                                              std=[0.210, 0.210, 0.210]) ])
        self._cuda=True        #GPU = True, CPU = False
        
        #variables not to edit here. You Can edit path in PathManagement Class.
        self.dataset_metadata, self.data_root_dir = pathManagement.dataPath(dataPlace = self.data_place, 
                                                                            dataType = "generated", 
                                                                            isNoise = False)

        self.dataset_metadata_test, self.data_root_dir_test = pathManagement.dataPathTest()
        
        #additional
        self.debug_mode = False
    
    def device(self):
        if self._cuda == True:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            return "cpu"
        