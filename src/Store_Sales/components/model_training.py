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


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.train = np.load(CLEAN_TRAIN_ARRAY_PATH)
        self.valid = np.load(CLEAN_VALID_ARRAY_PATH)
        self.secrets=read_yaml(SECRETS_FILE_PATH)
        DAGSHUB_USER_NAME=self.secrets.DAGSHUB_USER_NAME
        DAGSHUB_REPO_NAME=self.secrets.DAGSHUB_REPO_NAME
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.secrets.DAGSHUB_TOKEN
        os.environ['MLFLOW_TRACKING_URI']=f"https://dagshub.com/{DAGSHUB_USER_NAME}/{DAGSHUB_REPO_NAME}.mlflow"
    
    
    def get_X_y_from_train(self):
        try:
            X_train,y_train=self.train[:,:-1],self.train[:,-1]        
            return X_train,y_train
        except Exception as e:
            raise e
    
    def get_X_y_from_valid(self): 
        try:       
            X_test,y_test=self.valid[:,:-1],self.valid[:,-1]
            return X_test,y_test
        except Exception as e:
            raise e
    
    def basic_regressor(self):
        try:
            from sklearn.linear_model import LinearRegression
            
            params=self.model_tuning_and_get_parameters(LinearRegression())
            logger.info('Fetched best parms from model_tuning')
            regressor = LinearRegression(copy_X = params['copy_X'], 
                        fit_intercept = params['fit_intercept'], 
                        positive=  params['positive'],
                        n_jobs=params['n_jobs'])
        
            return regressor,params
        except Exception as e:
            raise e
    
    def random_regressor(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            params=self.model_tuning_and_get_parameters(RandomForestRegressor())
            logger.info('Fetched best parms from model_tuning')        
            regressor = RandomForestRegressor(n_estimators=params['n_estimators'],
                            max_depth=params['max_depth'],
                            criterion=params['criterion'],
                            random_state=params['random_state'],
                            bootstrap=params['bootstrap'],
                            max_features=params['max_features'],
                            min_samples_split=params['min_samples_split'],
                            min_samples_leaf=params['min_samples_leaf'])                            
            return regressor,params
        except Exception as e:
            raise e

    def get_train_time_n_accuracy(self,model,X_train,y_train):
        try:
            train_start = time.time()  
            model.fit(X_train,y_train) 
            train_end = time.time()
            eval_time_train = round(train_end-train_start,4)   
            train_accuracy=model.score(X_train, y_train)    
            return eval_time_train,train_accuracy
        except Exception as e:
            raise e
    
        
    def predict_on_test(self,model,X_test):
        try:
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_end = time.time()
            eval_time_predict = round(predict_end-predict_start,4)  
            logger.info('Prediction done by model on test data.')  
            return y_pred,eval_time_predict
        except Exception as e:
            raise e
    
    def get_metrics(self,y_true,y_pred,train_time,prediction_time,train_accuracy):
        try:
            from sklearn.metrics import r2_score,mean_squared_error
            acc = r2_score(y_true, y_pred)
            error=mean_squared_error(y_true, y_pred, squared=False)
            logger.info('Model evaluation metrics generated.') 
            return {'train_time':train_time,'train_accuracy':round(train_accuracy,2),'prediction_time':prediction_time,'prediction_accuracy': round(acc, 2), 'error': round(error, 2)}
        except Exception as e:
            raise e
      
        
    def get_experiment_id(self,name):
        try:
            exp = mlflow.get_experiment_by_name(name)
            if exp is None:
                exp_id = mlflow.create_experiment(name)
                return exp_id
            return exp.experiment_id
        except Exception as e:
            raise e
    
    def create_experiment(self,id,exp_name,model,train_time,train_accuracy,name,params=None):
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
            logger.info('Mlflow experiment starts')
            run_name=exp_name+"_"+name+"_"+str(datetime.now().strftime("%d-%m-%y:%H-%M-%S"))
            with mlflow.start_run(experiment_id=id,run_name=run_name):
                X_test,y_test=self.get_X_y_from_valid()           
                y_pred,prediction_time=self.predict_on_test(model,X_test)
                metrics=self.get_metrics(y_test,y_pred,train_time,prediction_time,train_accuracy)
                
                if not params == None:
                    for param in params:
                        mlflow.log_param(param, params[param])
                for metric in metrics:
                    mlflow.log_metric(metric, metrics[metric])          
                mlflow.sklearn.log_model(model, "model",registered_model_name=name)
            logger.info('Mlflow experiment ends')
        except Exception as e:
            raise e

         
    

    def initiate_experimentation(self,regression_func,exp_name,name):
        try:
            id=self.get_experiment_id(exp_name)
            logger.info(f'Experiment id = {id}, Experiment name= {exp_name}')
            X_train,y_train=self.get_X_y_from_train()
            model,params=regression_func()
            logger.info(f'\nModel : {model}')
            train_time,train_accuracy=self.get_train_time_n_accuracy(model,X_train,y_train) 
            logger.info(f'Training time and accuray generated')       
            self.create_experiment(id=id,exp_name=exp_name,model=model,train_time=train_time,
                                train_accuracy=train_accuracy,name=name,params=params)
        except Exception as e:
            raise e
    
    def initiate_mlflow_experimentations(self):
        try:
            #self.initiate_experimentation(regression_func=self.basic_regressor,exp_name='first_trials',name='LRs')
            #self.initiate_experimentation(regression_func=self.random_regressor,exp_name='trial_2',name='RFR')
            #self.initiate_experimentation(regression_func=self.basic_regressor,exp_name='trials_3',name='LRs')
            #self.initiate_experimentation(regression_func=self.random_regressor,exp_name='trials_3',name='RFR')
            #self.initiate_experimentation(regression_func=self.random_regressor,exp_name='trials_4_ohe',name='LRs')
            #self.initiate_experimentation(regression_func=self.random_regressor,exp_name='trials_4_ohe',name='RFR')
            #self.initiate_experimentation(regression_func=self.basic_regressor,exp_name='trials_5_ohe',name='LRs')
            
            self.initiate_experimentation(regression_func=self.random_regressor,exp_name='trials_5_ohe',name='RFR')
        except Exception as e:
            raise e
    
    
    def save_mlflow_experiments(self):
        try:
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
            client = mlflow.tracking.MlflowClient()
            all_experiments=[exp.experiment_id for exp in mlflow.search_experiments()]
            runs = mlflow.search_runs(experiment_ids=all_experiments)

            runs_file_path=os.path.join(self.config.root_dir,self.config.experiment_runs_filename+".csv")
            column_list=['run_id', 'experiment_id', 'status',  'start_time',
                        'end_time', 'metrics.train_time', 'metrics.train_accuracy',
                        'metrics.prediction_time','metrics.prediction_accuracy',
                        'metrics.error', 'tags.mlflow.runName', 'tags.mlflow.user','artifact_uri']       
            runs=runs[column_list]
            runs.to_csv(runs_file_path,index=False)
            logger.info(f'{self.config.experiment_runs_filename}.csv saved at {self.config.root_dir}')
            #print(runs.columns)
        except Exception as e:
            raise e

    def model_tuning_and_get_parameters(self,regression_func):
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV            
            
            if str(regression_func)=='LinearRegression()':
                parameters=  {
                            'copy_X' : [True,False],
                            'n_jobs' : [1,5,10,25,50,100],
                            'positive': [True,False],
                            'fit_intercept': [True,False]}
            else:
                parameters = {
                        "n_estimators":[50,100,120],
                        #"n_estimators":[120],
                        #"max_features":[0.6],
                        #"max_depth":[8],                       
                        #"criterion":['poisson'],
                        #"bootstrap":[True],
                        #"random_state":[42],
                        #"min_samples_split":[5],
                        "min_samples_leaf":[1,2,5],
                        "max_features":[0.3,0.6,1],
                        "max_depth":[2,4,8,16,18],
                        "min_samples_split":[2,5,10],
                        #"min_samples_leaf":[],
                        "criterion":['squared_error','poisson'],
                        "bootstrap":[True,False],
                        "random_state":[0,10,42]}
            


            model=regression_func   
            
            
            X_train,y_train=self.get_X_y_from_train()
            #scoring = {"r2_score": "r2", "mse":'neg_mean_squared_error'}
            CV_model = GridSearchCV(estimator=model, param_grid=parameters,scoring="r2",cv= 5) #refit="r2_score")
            CV_model.fit(X_train, y_train)
            #print(f'Best parameters : {CV_model.best_params_}')
            #print(f'Best score : {CV_model.best_score_}')
            return CV_model.best_params_

        except Exception as e:
            raise e