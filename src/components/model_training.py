import os
import sys

import joblib
import bentoml
from torch._dynamo.logging import pbar
from tqdm import tqdm

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Optimizer

from src.logger import logging
from src.exception import CustomException
from src.model_arch.arch import CNN
from src.constant.training_pipeline import *
from  src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact


class ModelTrainer:
    def __init__(
            self, data_transformation_artifact:DataTransformationArtifact,
            model_training_config:ModelTrainingConfig
        ): 
        
        self.model_training_config = model_training_config
        self.data_transformation_artifact = data_transformation_artifact

        self.model:Module = CNN()

        
    def train(self, optimizer:Optimizer)->None:
            """
            Description: To train the model

            input: model,device,train_loader,optimizer,epoch

            output: loss, batch id and accuracy
            """

            logging.info('Entered the train method of Model_trainer class')

            try:
                    self.model.train()

                    pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

                    correct:int = 0
                    processed=0

                    for batch_idx, (data, target) in enumerate(pbar):

                        # Initialization of gradient
                        optimizer.zero_grad()

                        # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                        # or specific requirements
                        ## prediction on data

                        y_pred = self.model(data)
                        
                        # Calculating loss given the prediction
                        loss = F.nll_loss(y_pred, target)

                        # Back Propogation
                        loss.backward()

                        # get the index of the log-probability corresponding to the max value
                        pred = y_pred.argmax(dim=-1, keepdim=True)

                        correct += pred.eq(target.view_as(pred)).sum().item()

                        processed += len(data)

                        pbar.set_description(
                            desc=f'Loss={loss.item()}, Batch_id={batch_idx}, Accuracy={100*correct/processed:0.2f}'
                        )

                    logging.info('Exited the train method of Model trainer class')

            except Exception as e:
                raise CustomException(e, sys)


    def test(self)->None:
        
            try:

                """
            Description: To test the model

            input: model, DEVICE, test_loader

            output: average loss and accuracy

            """
                logging.info('Entered the test method of Model trainer class')

                self.model.eval()

                test_loss:float = 0.0

                correct:int = 0

                with torch.no_grad():
                    for data, target in self.data_transformation_artifact.transformed_test_object:

                        data, target = data.to(DEVICE), target.to(DEVICE)

                        output = self.model(data)

                        test_loss += F.nll_loss(output, target, reduction='sum').item()

                        pred = output.argmax(dim=1, keepdim=True)

                        correct += pred.eq(target.view_as(pred)).sum().item()

                    test_loss /= len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    )

                    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0
                        * correct
                        / len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        )
                    )

                logging.info(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    correct,
                    len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                    100.0
                    * correct
                    / len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                    )
                )

                logging.info('Exited the test method of Model trainer class')

            except Exception as e:
                raise CustomException(e, sys)
                        










