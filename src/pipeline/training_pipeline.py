import sys

from src.components import data_transformation
from src.components import model_training
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.artifact_entity import (DataIngestionArtifact, DataTransformationArtifact, 
                                        ModelTrainingArtifact, ModelEvalArtifact)

from src.entity.config_entity import (DataIngestionConfig, DataTransformationConfig,
                                      ModelTrainingConfig, ModelEvalConfig)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_training_config = ModelTrainingConfig()
        self.model_eval_config = ModelEvalConfig()


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

        
    def start_data_transformation(self,
        data_ingestion_artifact:DataIngestionArtifact) -> DataTransformationArtifact:

        logging.info('Entered the start_data_transformation method of TrainPipeline class')

        try:
                data_transformation = DataTransformation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    data_transformation_config=self.data_transformation_config
                    )

                data_transformation_artifact = data_transformation.initiate_data_transformation()

                logging.info('Exited the start_data_transformation method of TrainPipeline class')

                return data_transformation_artifact

        except Exception as e:
                raise CustomException(e, sys)


    def start_model_training(
        self, data_transformation_artifact:DataTransformationArtifact)-> ModelTrainingArtifact:

        logging.info('Entered the start_model_training method of TrainPipeline class')

        try:

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_training_config=self.model_training_config
            )

            model_training_artifact = model_trainer.initiate_model_training()

            logging.info('Exited the start_model_training method of TrainPipeline class')

            return model_training_artifact

        except Exception as e:
            raise CustomException(e, sys)

    
    def start_model_evaluation(
        self, model_training_artifact:ModelTrainingArtifact,
        data_transformation_artifact:DataTransformationArtifact
        ) -> ModelEvalArtifact:

        logging.info('Entered the start_model_evaluation method of TrainPipeline class')

        try:
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_evaluation_config = self.model_eval_config,
                model_training_artifact=model_training_artifact 
            )

            model_eval_artifact = model_evaluation.initiate_model_evaluation()

            logging.info('Exited the start_model_evaluation method of TrainPipeline class')

            return model_eval_artifact

        except Exception as e:
            raise CustomException(e, sys)




    def run_pipeline(self)->None:
        logging.info('Entered the run_pipeline method of TrainPipeline class')

        try:
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()

            data_transformation_artifact:DataTransformationArtifact = (
                self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact))

            model_training_artifact:ModelTrainingArtifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact
            )

            model_eval_artifact:ModelEvalArtifact = self.start_model_evaluation(
                model_training_artifact=model_training_artifact,
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info('Exited the run_pipeline method of TrainPipeline class')

        except Exception as e:
            raise CustomException(e, sys)