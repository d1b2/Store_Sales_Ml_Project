import os
import sys
from Store_Sales.entity import *
from Store_Sales.utils import *
from Store_Sales import logger
from Store_Sales.constants import *
from pathlib import Path
import numpy as np
import pandas as pd
import time
import mlflow
from datetime import datetime
import pickle


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.exp_df=pd.read_csv(EXPERIMENTS_FILE_PATH)
        self.secrets=read_yaml(SECRETS_FILE_PATH)
        DAGSHUB_USER_NAME=self.secrets.DAGSHUB_USER_NAME
        DAGSHUB_REPO_NAME=self.secrets.DAGSHUB_REPO_NAME
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.secrets.DAGSHUB_TOKEN
        os.environ['MLFLOW_TRACKING_URI']=f"https://dagshub.com/{DAGSHUB_USER_NAME}/{DAGSHUB_REPO_NAME}.mlflow"
    
 

    def get_model_version_name_df(self):
        try:        
            client = mlflow.tracking.MlflowClient()
            run_ids=self.exp_df['run_id']     

            # Get the version of the model filtered by run_ids
            run_id_1,name,versions,current_stages=[],[],[],[]
            for i in range(len(run_ids)):
                result=list(client.search_model_versions("run_id='{}'".format(run_ids[i])))
                for res in result:                    
                    run_id_1.append(res.run_id)
                    name.append( res.name)
                    versions.append(int(res.version))
                    current_stages.append(res.current_stage)                
            
            version_name_df=pd.DataFrame({
                'run_id' : run_id_1,
                'name' : name,
                'version' : versions,
                'current_stage' :current_stages})
            
                      
            return version_name_df        
        except Exception as e:
            raise e
    

    def merge_version_name_exp_df(self):
        
        try:
            version_name_df=self.get_model_version_name_df()
            merged_df=self.exp_df.merge(version_name_df, on='run_id')
            merged_df['Selected']= [True if merged_df['metrics.train_accuracy'][i]>self.config.base_accuracy and \
                                        merged_df['metrics.prediction_accuracy'][i]>self.config.base_accuracy else False\
                                        for i in range (len(merged_df))]
            return merged_df 
        except Exception as e:
            raise e          


      
    def model_staging(self,dataframe,stage):
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
            client = mlflow.tracking.MlflowClient()
            for i in range(len(dataframe)):
                client.transition_model_version_stage(
                                    name=dataframe['name'][i],
                                    version=dataframe['version'][i],
                                    stage=stage)
            logger.info(f'{len(dataframe)} Models send to {stage}.')
        except Exception as e:
            raise e
        
    def initiate_model_staging(self,dataframe):
        try:            
            staging_df=dataframe[dataframe['Selected']==True]
            staging_df.reset_index(drop=True,inplace=True)
            self.model_staging(dataframe=staging_df,stage='Staging')

            none_df=dataframe[dataframe['Selected']!=True]
            none_df.reset_index(drop=True,inplace=True)            
            self.model_staging(dataframe=none_df,stage='None')
            
            production_df = staging_df[
                (staging_df['metrics.train_accuracy']-staging_df['metrics.prediction_accuracy'] < 0.1) & \
                (staging_df['metrics.error']==min(staging_df['metrics.error']))]
            
            production_df.reset_index(drop=True,inplace=True)
            self.model_staging(dataframe=production_df,stage='Production')         
        except Exception as e:
            raise e      
   

    def download_production_model_file(self):
        try:
            merged_df=self.merge_version_name_exp_df()
            production_run_id=merged_df[(merged_df['current_stage']=='Production')].run_id.values[0]           
            merged_df_path=os.path.join(self.config.root_dir,self.config.model_filename)
            merged_df.to_csv(merged_df_path)
            logger.info('Model evaluation csv saved.')
            mlflow.artifacts.download_artifacts(run_id=production_run_id,dst_path=self.config.root_dir)
            logger.info('Model files in production stage downloaded.')          
        except Exception as e:
            raise e
        
    def initiate_model_evaluation(self):
        try:
            logger.info(f"\n\n{'='*20} Model Evaluation log started. {'='*20} \n\n") 
            merged_df=self.merge_version_name_exp_df()
            self.initiate_model_staging(merged_df)
            self.download_production_model_file()
            logger.info(f"\n\n{'='*20} Model Evaluation log completed. {'='*20} \n\n") 
        except Exception as e:
            raise e
