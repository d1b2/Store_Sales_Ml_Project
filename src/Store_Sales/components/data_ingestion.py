import os
from Store_Sales.entity import *
from Store_Sales.utils import *
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
        return dataset  
    
    
    def get_kaggle_username_key(self):
        with open(self.config.kaggle_file_path) as user:
            user_info=json.load(user)
        return user_info
    
    def download_kaggle_dataset(self,dataset,user_info):
        
        os.environ['KAGGLE_USERNAME'] = user_info['username']
        os.environ['KAGGLE_KEY'] =   user_info['key'] 

        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_file(dataset, self.config.test_filename, self.config.root_dir)
        api.dataset_download_file(dataset, self.config.train_filename, self.config.root_dir)

        test_file_path = os.path.join(self.config.root_dir,self.config.test_filename).replace("\\","/")
        train_file_path = os.path.join(self.config.root_dir,self.config.train_filename).replace("\\","/")

        data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
        logger.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
        return data_ingestion_artifact
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
         try:
            dataset=self.dataset_name()
            user_info=self.get_kaggle_username_key()
            self.download_kaggle_dataset(dataset=dataset,user_info=user_info)
         except Exception as e:
            raise e
    
    def __del__(self):
        logger.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
    


    
