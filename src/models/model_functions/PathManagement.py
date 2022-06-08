class PathManagement:
    def __init__(self):
        self.__cloud_path_prefix = "/content/drive/My Drive/"
        #########___Data PATH___##############
        #FOR LOCAL DATA:
        #--REAL DATA:
        self._localDataset_metadata = "../../data/raw/1channel/reference/training/epsilon.csv"
        self._localData_root_dir = "../../data/raw/1channel/photo/training/"
        #--SELF-GENERATED DATA:
        #----UNNOISED
        self._localData_metadata_generated_unnoised = "../../data/generated/unnoised/reference/training/epsilon.csv"
        self._localData_root_dir_generated_unnoised = "../../data/generated/unnoised/photo/training/"
        #----NOISED
        self._localData_metadata_generated_noised = "../../data/generated/noised/reference/training/epsilon.csv"
        self._localData_root_dir_generated_noised = "../../data/generated/noised/photo/training/"
        #----MIXED
        self._localData_metadata_generated_mixed = "../../data/generated/mixed/reference/training/epsilon.csv"
        self._localData_root_dir_generated_mixed = "../../data/generated/mixed/photo/training/"

        #--TEST
        #----NOISED
        # self._localData_metadata_test = "../../data/generated/noised/reference/test/epsilon.csv"
        # self._localData_root_dir_test = "../../data/generated/noised/photo/test"
        #----UNNOISED
        # self._localData_metadata_test = "../../data/generated/unnoised/reference/test/epsilon.csv"
        # self._localData_root_dir_test = "../../data/generated/unnoised/photo/test/"
        #----REALDATA
        self._localData_metadata_test = "../../data/raw/1channel/reference/test/epsilon.csv"
        self._localData_root_dir_test = "../../data/raw/1channel/photo/test/"

        #ON DRIVE:
        #--REAL DATA:
        self._cloudDataset_metadata = self.__cloud_path_prefix + "data/reference/real/epsilon_short.csv"
        self._cloudData_root_dir = self.__cloud_path_prefix + "data/photo/real/"
        #--SELF-GENERATED DATA:
        #----UNNOISED
        self._cloudData_metadata_generated_unnoised = self.__cloud_path_prefix + "data/reference/generated/unnoised/epsilon.csv"
        self._cloudData_root_dir_generated_unnoised = self.__cloud_path_prefix + "data/photo/generated/unnoised/"
        #----NOISED
        self._cloudData_metadata_generated_noised = self.__cloud_path_prefix + "data/reference/generated/noised/epsilon.csv"
        self._cloudData_root_dir_generated_noised = self.__cloud_path_prefix + "data/photo/generated/noised/"


        #########___Model PATH___##############
        self.__path_save_model_cloud = self.__cloud_path_prefix + "data/models/"
        self.__path_save_model_local = "../../models/"

    def dataPath(self, dataPlace = "local", dataType = "original", isNoise = True):
        """! define correct data path using parameters
        
        @param dataPlace  data place can be 'local' or 'cloud'.
        @param dataType   data type can be 'original' or 'generated' or 'mixed'.
        @param isNoise    only used in case of generated dataType.

        @return 2 path --> 1. with methadata, 2. with photo
        """
        if dataPlace == 'local':
            if dataType == 'original':
                return self._localDataset_metadata, self._localData_root_dir
            elif dataType == 'generated':
                if isNoise == False:
                    return self._localData_metadata_generated_unnoised, self._localData_root_dir_generated_unnoised
                else:
                    return self._localData_metadata_generated_noised, self._localData_root_dir_generated_noised
            elif dataType == 'mixed':
                return self._localData_metadata_generated_mixed, self._localData_root_dir_generated_mixed
            else:
                return False
        elif dataPlace == 'cloud':
            if dataType == 'original':
                return self._cloudDataset_metadata, self._cloudData_root_dir
            elif dataType == 'generated':
                if isNoise == False:
                    return self._cloudData_metadata_generated_unnoised, self._cloudData_root_dir_generated_unnoised
                else:
                    return self._cloudData_metadata_generated_noised, self._cloudData_root_dir_generated_noised
            else:
                return False
        else:
          return False

    def dataPathTest(self):
        return self._localData_metadata_test, self._localData_root_dir_test

    def modelSavePath(self, dataPlace = "local"):
        """! define model save path depending on the save location
        
        @param dataPlace  data place can be 'local' or 'cloud'.

        @return model save path
        """
        if dataPlace == "local":
            return self.__path_save_model_local
        elif dataPlace == "cloud":
            return self.__path_save_model_cloud
        else: return False