import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.constant.training_pipeline import *
from src.cloud_storage.s3_operation import S3Operation
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()

    def get_data_from_s3(self) -> None:
        try:
            logging.info('Initiated the get_data_from_s3 method of DataIngestion class')

            self.s3.sync_folder_from_s3(
                folder = self.data_ingestion_config.data_path,
                bucket_name = self.data_ingestion_config.bucket_name,
                bucket_folder_name = self.data_ingestion_config.s3_data_folder
            )
        except Exception as e:
            raise CustomException(e,sys)

        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        
        logging.info('Starting the DataIngestion')

        try:
            self.get_data_from_s3()

            data_ingestion_artifact:DataIngestionArtifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.train_data_path,
                test_file_path = self.data_ingestion_config.test_data_path
            )

            logging.info('DataIngestion Completed')

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) 


