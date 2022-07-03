import logging
import os.path

logging.basicConfig(
    level = logging.ERROR,
    format = "{asctime} {levelname:<8} {message}",
    style = '{',
    #filename='pathManagement.log',
    filemode='w')

class PathManagement:
    #variables
    __metadata_file_name = "epsilon.csv"
    __path_save_model = "../../models/"

    def __init__(self, dataType = "original", noiseType = "noised", centerInTheMiddle = False, purposeData = "training"):
        """initialize all data for path.

        Args:
            dataType (str, optional): The origin of the data. Can be "original" - data from camera and "generated" - data from generator. Defaults to "original". Possible "original", "generated"
            noiseType (str, optional): Data can be with noise, without noise and mixed - some data with generated noise, and some data without. Defaults to "noised". Possible "noised", "unnoised", "mixed"
            centerInTheMiddle (bool, optional): In generated data case which data to use. With the center in center of the photo or not. Defaults to False.
            purposeData (str, optional): Purpose of data. For training will be use another data than for test. Defaults to "training". Possible "training", "test"
        """
        self.__path_location = "../../data/"
        self.__dataset_metadata = "" #metadata in csv
        self.__data_root_dir = "" #photos

        self.__dataType = dataType 
        self.__noiseType = noiseType 
        self.__centerInTheMiddle = centerInTheMiddle 
        self.__purposeData = purposeData 
        
        
        self.__createPathToData(filename = self.__metadata_file_name)

    #public method
    def isPathExist(self):
        """this function is used to check if paths with metadata and photos exist 
        """
        if not os.path.exists(path=self.__data_root_dir):
            logging.error("data_root_dir path doesn't exist")
            return False
        elif not os.path.exists(path=self.__dataset_metadata):
            logging.error("dataset_metadata path doesn't exist")
            return False
        else:
            return True

    def setTemporaryPaths(self, dataset_metadata, data_root_dir):
        """is used to set paths with metadata and photos

        Args:
            dataset_metadata (str): path to metadata - directly to csv file
            data_root_dir (str): path to photos
        """
        self.__dataset_metadata = dataset_metadata
        self.__data_root_dir = data_root_dir
        self.isPathExist()

    def changeMetadataFileName(self, filename = "epsilon.csv"):
        """change file name.

        Args:
            filename (str, optional): file name to change. Defaults to "epsilon.csv".
        """
        self.__metadata_file_name = filename      
        self.__init__(dataType=self.__dataType,noiseType=self.__noiseType,centerInTheMiddle=self.__centerInTheMiddle,purposeData=self.__purposeData)
        self.isPathExist()

    def getDataPath(self):
        """return path to metadata (CSV) and photos

        Returns:
            str: metadata path
            str: photo path
        """
        return self.__dataset_metadata, self.__data_root_dir

    @staticmethod
    def getModelSavePath():
        return PathManagement.__path_save_model


    #private methods
    def __createMainDataPath(self):
        if self.__dataType == "generated":
            self.__path_location += "generated/"
            self.__selectNoiseType()
            self.__isInTheMiddle()
        elif self.__dataType == "original":
            self.__path_location += "raw/1channel/"
        else:
            logging.error("Wrong data type argument!!!")

    def __createPathToData(self, filename):
        self.__createMainDataPath()

        self.__data_root_dir = self.__path_location + "photo/"
        self.__dataset_metadata = self.__path_location + "reference/"
        self.__addPurposeData()
        self.__dataset_metadata += filename
        self.isPathExist()
    
    def __addPurposeData(self):
        self.__data_root_dir += self.__purposeData +"/"
        self.__dataset_metadata += self.__purposeData + "/"

    def __selectNoiseType(self):
        if self.__noiseType == "noised" or self.__noiseType == "unnoised" or self.__noiseType == "mixed":
            self.__path_location += self.__noiseType
        else:
            logging.error("Wrong noise type argument!!!")

    def __isInTheMiddle(self):
        if self.__centerInTheMiddle == True:
            self.__path_location += "_middle/"
        else:
            self.__path_location += "/"
