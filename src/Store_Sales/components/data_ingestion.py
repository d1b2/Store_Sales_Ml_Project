import os
from Store_Sales.entity import *
from Store_Sales.utils import *
from Store_Sales import logger
from pathlib import Path
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
#import numpy as np
import pandas as pd
import re

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def dataset_name(self):
        kaggle_url = re.split(r'/', self.config.source_URL)
        dataset ="/".join(kaggle_url[-2:])
        logger.info(f"Dataset name :[{dataset}] extracted from url: {self.config.source_URL}")
        return dataset  
    
    
    def get_kaggle_username_key(self):
        with open(self.config.kaggle_file_path) as user:
            user_info=json.load(user)
        logger.info(f"Getting user info from {self.config.kaggle_file_path}")
        return user_info
    
    def download_kaggle_dataset(self,dataset,user_info):
        
        os.environ['KAGGLE_USERNAME'] = user_info['username']
        os.environ['KAGGLE_KEY'] =   user_info['key'] 

        api = KaggleApi()
        api.authenticate()
        logger.info("Authenticating Kaggle Api with user info.")
        api.dataset_download_file(dataset, self.config.test_filename, self.config.root_dir)
        api.dataset_download_file(dataset, self.config.train_filename, self.config.root_dir)
        logger.info(f"Downloading Test and Train dataset at {self.config.root_dir}.")

       
    
    def initiate_data_ingestion(self):
         try:
            logger.info(f"\n\n{'='*20}Data Ingestion log started.{'='*20} \n\n")
            dataset=self.dataset_name()
            user_info=self.get_kaggle_username_key()
            self.download_kaggle_dataset(dataset=dataset,user_info=user_info)
            logger.info(f"\n\n{'='*20}Data Ingestion log completed.{'='*20} \n\n")
         except Exception as e:
            raise e
    
    #def __del__(self):
        #logger.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
    


    
