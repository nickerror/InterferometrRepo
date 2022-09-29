import os
import pandas as pd
import cv2

class ImageManager:
    """datatype: generatedUnnoised or generatedNoised or real"""
    def __init__(self, dataType):
        self.__cloud_path_prefix = "/content/drive/My Drive/"
        #########___Data PATH___##############
        #FOR LOCAL DATA:
        #--REAL DATA:
        self._localDataset_metadata = "data/raw/1channel/reference/epsilon.csv"
        self._localData_root_dir = "data/raw/1channel/photo/"
        #--REAL DATA Filtered:
        self._localDataset_metadata_filtered = "data/raw/1channel/reference/epsilon.csv"
        self._localData_root_dir_filtered = "data/raw/1channel/medianedPhoto/"
        #--SELF-GENERATED DATA:
        #----UNNOISED
        self._localData_metadata_generated_unnoised = "data/generated/unnoised/reference/epsilon.csv"
        self._localData_root_dir_generated_unnoised = "data/generated/unnoised/photo/"
        #----UNNOISED Middle
        self._localData_metadata_generated_unnoisedMiddle = "data/generated/unnoisedMiddle/reference/epsilon.csv"
        self._localData_root_dir_generated_unnoisedMiddle = "data/generated/unnoisedMiddle/photo/"
        #----NOISED
        self._localData_metadata_generated_noised_middle = "data/generated/noisedMiddle/reference/epsilon.csv"
        self._localData_root_dir_generated_noised_middle = "data/generated/noisedMiddle/photo/"
        #----NOISED MIDDLE
        self._localData_metadata_generated_noised = "data/generated/noised/reference/epsilon.csv"
        self._localData_root_dir_generated_noised = "data/generated/noised/photo/"
        #--MIXED-DATA:
        self._localData_metadata_generated_mixed = "data/mixed/reference/epsilon.csv"
        self._localData_root_dir_generated_mixed = "data/mixed/photo/"
        #--MIXED-Filtered:
        self._localData_metadata_generated_mixed_filtered = "data/mixed/reference/epsilon.csv"
        self._localData_root_dir_generated_mixed_filtered = "data/mixed/medianedPhoto/"

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

        if(dataType=="generatedUnnoised"):
            self.root_dir=self._localData_root_dir_generated_unnoised
            self.annotation_file=self._localData_metadata_generated_unnoised
        elif(dataType=="generatedUnnoisedMiddle"):
            self.root_dir=self._localData_root_dir_generated_unnoisedMiddle
            self.annotation_file=self._localData_metadata_generated_unnoisedMiddle
        elif(dataType=="generatedNoised"):
            self.root_dir=self._localData_root_dir_generated_noised
            self.annotation_file=self._localData_metadata_generated_noised
        elif(dataType=="generatedNoisedMiddle"):
            self.root_dir=self._localData_root_dir_generated_noised_middle
            self.annotation_file=self._localData_metadata_generated_noised_middle
        elif(dataType=="real"):
            self.root_dir=self._localData_root_dir
            self.annotation_file=self._localDataset_metadata
        
        self.root_dir = self.root_dir
        self.annotations = pd.read_csv(self.annotation_file,skiprows=0, delim_whitespace=' ')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path=os.path.join(self.root_dir, str(str("%05d" %self.annotations.imgnr[index]))+ ".png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        epsRead = self.annotations.eps[index]
        return img, epsRead