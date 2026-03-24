import os
import sys
import joblib
from typing import Tuple

from src.logger import logging
from src.exception import CustomException

from torchvision import transforms
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader, Dataset

from src.entity.config_entity import  DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(
        self, data_ingestion_artifact:DataIngestionArtifact, 
        data_transformation_config:DataTransformationConfig):

        self.data_ingestion_artifact = data_ingestion_artifact

        self.data_transformation_config = data_transformation_config


    def transforming_train_data(self)-> transforms.Compose:
            try:
                logging.info('Entered the transforming_training_data method of Data transformation class')

                transformed_train:transforms.Compose = transforms.Compose([
                    transforms.Resize(self.data_transformation_config.resize),
                    transforms.CenterCrop(self.data_transformation_config.center_crop),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(self.data_transformation_config.random_rotation),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.data_transformation_config.normalize_transforms)
                ])

                logging.info('Exited the transforming_training_data method of Data transformation class')

                return transformed_train

            except Exception as e:
                raise CustomException(e, sys)


    def transforming_test_data(self) -> transforms.Compose:
            try:
                logging.info('Entered the transforming_testing_data method of Data transformation class')

                transformed_test:transforms.Compose = transforms.Compose([
                    transforms.Resize(self.data_transformation_config.resize),
                    transforms.CenterCrop(self.data_transformation_config.center_crop),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.data_transformation_config.normalize_transforms)
                ])

                logging.info('Exited the transforming_testing_data method of Data transformation class')

                return transformed_test

            except Exception as e:
                raise CustomException(e, sys)


    def data_loader(
        self, transformed_train:transforms.Compose, transformed_test:transforms.Compose
            )-> Tuple[DataLoader, DataLoader]:

            try: 
                logging.inf0('Entered the data_loader method of Data transformation class')

                train_data:Dataset = ImageFolder(
                    os.path.join(self.data_ingestion_artifact.train_file_path),
                    transform=transformed_train
                )

                test_data:Dataset = ImageFolder(
                    os.path.join(self.data_ingestion_artifact.test_file_path),
                    transform=transformed_test
                )

                logging.info('Created train data and test data paths')

                train_loader:DataLoader = DataLoader(
                    train_data, **self.data_transformation_config.data_loader_params
                )

                test_loader:DataLoader = DataLoader(
                    test_data, **self.data_transformation_config.data_loader_params
                )

                logging.info('Existed the data_loader method of Data Transformation class')

                return train_loader, test_loader

            except Exception as e:
                raise CustomException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact: 

            try:
                logging.info('Entered the initiate_data_transformation method of Data transformation class')

                transformed_train:transforms.Compose = self.transforming_train_data()
                transformed_test:transforms.Compose = self.transforming_test_data()

                os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

                joblib.dump(
                    transformed_train, self.data_transformation_config.train_transforms_file
                )

                joblib.dump(
                    transformed_test, self.data_transformation_config.test_transforms_file
                )

                train_loader, test_loader = self.data_loader(
                     transformed_train=transformed_train, transformed_test=transformed_test
                )

                data_transformation_artifact:DataTransformationArtifact = DataTransformationArtifact(
                    transformed_train_object=train_loader,
                    transformed_test_object=test_loader,
                    train_transforms_file_path=self.data_transformation_config.train_transforms_file,
                    test_transforms_file_path=self.data_transformation_config.test_transforms_file
                )

                logging.info('Exited the initiate_data_transformation method of Data transformation class')

                return data_transformation_artifact

            except Exception as e:
                raise CustomException(e, sys)