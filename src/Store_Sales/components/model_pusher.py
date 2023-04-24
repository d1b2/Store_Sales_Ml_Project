import os
import sys
from Store_Sales.entity import *
from Store_Sales.utils import *
from Store_Sales import logger
from Store_Sales.constants import *
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import pickle


class ModelPusher:
    def __init__(self, config: ModelPusherConfig):
        self.config = config
        
    def __get_transformer_trained_model(self):
        try:
            transformer= pickle.load(open(COL_TRANSFORMER_FILE_PATH, 'rb'))
            model = pickle.load(open(TRAINED_MODEL_FILE_PATH, 'rb'))
            return transformer,model
        except Exception as e:
            raise e
    
    def __preliminary_test_on_model(self,transformer,model):
        try:
            test_df=pd.read_csv(CLEAN_TEST_FILE_PATH)        
            df=transformer.transform(test_df)
            predictions=model.predict(df)
            return len(predictions)==len(test_df)
        except Exception as e:
            raise e
    
    def __push_object(self,object_path):
        try:
            shutil.copy(src=object_path,dst=self.config.dest_dir)
            logger.info(f'{os.path.basename(object_path)} copied from {(object_path)} to {self.config.dest_dir}')
        except Exception as e:
            raise e

    
    def initiate_model_pusher(self):
        try:
            logger.info(f"\n\n{'='*20} Model Pusher log started. {'='*20} \n\n")
            transformer,model=self.__get_transformer_trained_model()
            preliminary_test_status=self.__preliminary_test_on_model(transformer,model)
            if preliminary_test_status==True:
                logger.info(f"Prelimnary test passed by trained model : {preliminary_test_status}")
                self.__push_object(COL_TRANSFORMER_FILE_PATH)
                self.__push_object(TRAINED_MODEL_FILE_PATH)
            else:
                logger.info(f"Prelimnary test passed by trained model : {preliminary_test_status}")
                logger.info("Model Pusher failed.")
            
            logger.info(f"\n\n{'='*20} Model Pusher log completed. {'='*20} \n\n") 
        except Exception as e:
            raise e


    
        