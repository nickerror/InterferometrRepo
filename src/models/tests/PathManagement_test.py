from importlib.resources import path
import pytest
import sys


sys.path.append(sys.path[0] + '\\..')

from model_functions.PathManagement import PathManagement

class Test_Generated_Noised:
    def test_roots_generated_noised_middle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="noised", centerInTheMiddle=True, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/noised_middle/photo/test/"
        assert dataset_metadata == "../../data/generated/noised_middle/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_noised_middle_training(self):
        pathManagement = PathManagement(dataType="generated", noiseType="noised", centerInTheMiddle=True, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/noised_middle/photo/training/"
        assert dataset_metadata == "../../data/generated/noised_middle/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_noised_notMiddle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="noised", centerInTheMiddle=False, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/noised/photo/test/"
        assert dataset_metadata == "../../data/generated/noised/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_noised_notMiddle_training(self):
        pathManagement = PathManagement(dataType="generated", noiseType="noised", centerInTheMiddle=False, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/noised/photo/training/"
        assert dataset_metadata == "../../data/generated/noised/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==True

class Test_Generated_Unnoised:
    def test_roots_generated_unnoised_middle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="unnoised", centerInTheMiddle=True, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/unnoised_middle/photo/test/"
        assert dataset_metadata == "../../data/generated/unnoised_middle/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_unnoised_middle_training(self):
        pathManagement = PathManagement(dataType="generated", noiseType="unnoised", centerInTheMiddle=True, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/unnoised_middle/photo/training/"
        assert dataset_metadata == "../../data/generated/unnoised_middle/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_unnoised_notMiddle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="unnoised", centerInTheMiddle=False, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/unnoised/photo/test/"
        assert dataset_metadata == "../../data/generated/unnoised/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_generated_unnoised_notMiddle_training(self):
        pathManagement = PathManagement(dataType="generated", noiseType="unnoised", centerInTheMiddle=False, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/unnoised/photo/training/"
        assert dataset_metadata == "../../data/generated/unnoised/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==True

class Test_Generated_Mixed:
    def test_roots_generated_mixed_notMiddle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="mixed", centerInTheMiddle=False, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/mixed/photo/test/"
        assert dataset_metadata == "../../data/generated/mixed/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==False #not exist in real. 

    def test_roots_generated_mixed_middle_training(self):
        pathManagement = PathManagement(dataType="generated", noiseType="mixed", centerInTheMiddle=True, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/mixed_middle/photo/training/"
        assert dataset_metadata == "../../data/generated/mixed_middle/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==False #not exist in real. 

    def test_roots_generated_mixed_middle_test(self):
        pathManagement = PathManagement(dataType="generated", noiseType="mixed", centerInTheMiddle=True, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/generated/mixed_middle/photo/test/"
        assert dataset_metadata == "../../data/generated/mixed_middle/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==False #not exist in real. 

class Test_Original:
    def test_roots_original_noised_middle_test(self):
        pathManagement = PathManagement(dataType="original", noiseType="noised", centerInTheMiddle=False, purposeData="test")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/raw/1channel/photo/test/"
        assert dataset_metadata == "../../data/raw/1channel/reference/test/epsilon.csv"
        assert pathManagement.isPathExist()==True

    def test_roots_original_noised_middle_training(self):
        pathManagement = PathManagement(dataType="original", noiseType="noised", centerInTheMiddle=False, purposeData="training")
        dataset_metadata, data_root_dir = pathManagement.getDataPath()
        assert data_root_dir == "../../data/raw/1channel/photo/training/"
        assert dataset_metadata == "../../data/raw/1channel/reference/training/epsilon.csv"
        assert pathManagement.isPathExist()==True