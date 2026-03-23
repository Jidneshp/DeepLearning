import sys

from src.constant import training_pipeline
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainPipeline

def start_pipeline():

    try:
        training_pipeline = TrainPipeline()

        training_pipeline.run_pipeline()

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    start_pipeline()
