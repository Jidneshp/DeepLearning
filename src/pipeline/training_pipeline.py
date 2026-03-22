import sys

from components import data_ingestion
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion

from src.entity.artifact_entity import (DataIngestionArtifact)

from src.entity.config_entity import (DataIngestionConfig)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info('Entered the start_data_ingestion method of TrainPipeline class')

        try:
            logging.info('getting data from S3 bucket')

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info('Got train and test data from S3')

            logging.info(
                'Exiting the start_data_ingestion method from TrainPipeline class'
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

