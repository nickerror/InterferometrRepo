from email.policy import default
import logging
import torch.utils
from model_functions.EpsilonDataset import EpsilonDataset
from model_functions.PathManagement import PathManagement
from model_functions.Config import Config
import os

def prepare_data(config, train = True):
    #create time logger:
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=50) #50 - critical, 40 - error, 30 - warning, 20 - info, 10 - debug, 0 - notset
    logging.debug('1. Start prepare_data')


    dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)
    #dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms)
    #dataset = EpsilonDataset(config.data_root_dir, config.dataset_metadata, transform=config.data_transforms_augmented)
    
    
    #g = torch.Generator(device=config.device()).manual_seed(23) 
    g = torch.Generator(device="cpu").manual_seed(23) 
    loader_params = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                            pin_memory=config.pin_memory, generator=g, shuffle=True)

    if train: #train and validation
        train_size = int(config.train_size * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

        print("len(train_dataset):", len(train_dataset),"len(val_dataset):", len(val_dataset))

        train_loader = torch.utils.data.DataLoader(**loader_params, dataset=train_dataset )
        validation_loader = torch.utils.data.DataLoader(**loader_params, dataset=val_dataset )
        
        return {'train': train_loader, 'val': validation_loader}
    else: #test only
        val_size = int(0)
        test_size = len(dataset)

        test_dataset, val_dataset = torch.utils.data.random_split(dataset, [test_size, val_size], generator=g)

        print("length test dataset:", len(test_dataset))

        test_loader = torch.utils.data.DataLoader(**loader_params, dataset=test_dataset )
        
        return {'test': test_loader}
    
def saveModel(model,config, model_name = "default"):
    """function to save model

    Args:
        model (ResNet): model to save
        config (Config): config object from Config class
        model_name (str, optional): name of model - prefered pith *.pth. If "default" then name from config. Defaults to "default".
    """

    if model_name == "default":
        tempPathToSave = PathManagement.getModelSavePath() + config.model_name_to_save #path to save
    else:
        tempPathToSave = PathManagement.getModelSavePath() + model_name #path to save

    torch.save(model, tempPathToSave)
    #torch.save(model.state_dict(), tempPathToSave)
    
    print("model saved: " + config.data_place)

def saveEpochModel(model,config, epoch_nr = 0, model_name = "default"):
    """function to save model after epoch

    Args:
        model (ResNet): model to save
        config (Config): config object from Config class
        epoch_nr (int, optional): epoch number
        model_name (str, optional): name of model - prefered pith *.pth. If "default" then name from config. Defaults to "default".
    """
    
    if model_name == "default":
        folderPathToSave = PathManagement.getModelSavePath() + config.model_name_to_save[:-4] + "-separate_epochs/"
        epochModelNameToSave = config.model_name_to_save[:-4] + "_epoch_" + str(epoch_nr) + ".pth"
        tempPathToSave =  folderPathToSave + epochModelNameToSave #path to save
    else:
        folderPathToSave =PathManagement.getModelSavePath() +  model_name[:-4] + "-separate_epochs/"
        epochModelNameToSave = model_name[:-4] + "_epoch_" + str(epoch_nr) + ".pth"
        tempPathToSave = folderPathToSave + epochModelNameToSave #path to save

    

    if not os.path.exists(folderPathToSave):
    
        # Create a new directory because it does not exist 
        os.makedirs(folderPathToSave)
        print("The new directory is created!")
  
    torch.save(model, tempPathToSave)
    
    print("model saved: " + config.data_place)
    print("model name: " + epochModelNameToSave)


def import_data_form_cloud(config ):
    #import data from google drive
    import matplotlib.image as mpimg
    # from google.colab import drive
    # drive.mount('/content/drive')
    from matplotlib import pyplot as plt
    

    #test connection:
    img = mpimg.imread(config.data_root_dir + '03400.png') #test display img')
    imgplot = plt.imshow(img)
    plt.show()

