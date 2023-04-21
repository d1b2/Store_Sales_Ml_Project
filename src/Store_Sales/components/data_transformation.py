import os
from Store_Sales.entity import *
from Store_Sales.utils import *
from Store_Sales.constants import *
from Store_Sales import logger
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,RobustScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import pickle


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.train=pd.read_csv(CLEAN_TRAIN_FILE_PATH)
        self.valid =pd.read_csv(CLEAN_VALID_FILE_PATH)
        
    def get_num_cat_columns(self):
        try: 
            #here 'Item_Visibility' is excluded from numerical columns since it contains some outliers        
            num_columns=[i for i in self.train.columns if self.train[i].dtype=='float' and i!='Item_Visibility']
            cat_columns=[i for i in self.train.columns if self.train[i].dtype=='object']           
            
            #to delete target feature from list
            num_columns.pop()
           
           
            logger.info("Lists of numerical and categorical features generated.")
            return num_columns,cat_columns
        except Exception as e:
            raise e
        
    def get_column_transformer(self):
        try:
            numerical_features,categorical_features=self.get_num_cat_columns()
            #for numerical features
            scaler=StandardScaler()
            #for Item_Visibility column
            robust=RobustScaler()
            #for categorical features
            #ord_encoder=OrdinalEncoder()
            one_hot_encoder=OneHotEncoder(drop='first', sparse=False)
            column_transformer_object = make_column_transformer(
                                    #(ord_encoder,categorical_features), 
                                    (one_hot_encoder,categorical_features),
                                    (robust,['Item_Visibility']),
                                    (scaler,numerical_features),
                                    remainder='passthrough')
            logger.info("Column transformer object generated.")
            return column_transformer_object
        except Exception as e:
            raise e
    
    def get_X_y_from_train_validation(self):
        try:
            X_train,y_train=self.train.iloc[:,:-1],self.train.iloc[:,-1]
            logger.info(f"Shape of X_train:{X_train.shape},Shape of y_train:{y_train.shape}")
            X_valid,y_valid=self.valid.iloc[:,:-1],self.valid.iloc[:,-1]
            logger.info(f"Shape of X_valid:{X_valid.shape},Shape of y_valid:{y_valid.shape}")
            return  X_train,y_train,X_valid,y_valid
        except Exception as e:
            raise e


    def save_object(self,path,object,message):
        try:
            np.save(path,object)
            logger.info(f'Saving transformed {message} array.')
        except Exception as e:
            raise e
    def save_column_transformer(self,path,object):
        try:
            pickle.dump(object, open(path, 'wb'))
            logger.info(f'Saving column transformer.')
            pass
        except Exception as e:
            raise e

    

    def initiate_data_transformation(self) :
        try:
            logger.info(f"\n\n{'='*20}Data Transformation log started.{'='*20} \n\n")            
                 
            column_transformer=self.get_column_transformer() 
            column_transformer_filepath=os.path.join(self.config.root_dir,self.config.column_transformer_object)
            self.save_column_transformer(column_transformer_filepath,column_transformer)
            X_train,y_train,X_valid,y_valid=self.get_X_y_from_train_validation()
            X_train_arr=column_transformer.fit_transform(X_train)
            print(X_train_arr.shape)
            logger.info("Fit_transform of column transformer applied to X_train.")
            train_arr= np.column_stack([X_train_arr, np.array(y_train)])
            logger.info("Fit_transformed X_train stacked to y_train.")
            train_arr_filepath=os.path.join(self.config.transform_dir,self.config.transform_train_filename +".npy")
            self.save_object(train_arr_filepath,train_arr,'train')
            X_valid_arr=column_transformer.transform(X_valid)
            print(X_valid_arr.shape)
            logger.info("Transform of column transformer applied to X_valid.")
            valid_arr= np.column_stack([X_valid_arr, np.array(y_valid)])
            logger.info("Transformed X_valid stacked to y_valid.")
            valid_arr_filepath=os.path.join(self.config.transform_dir,self.config.transform_validation_filename +".npy")
            self.save_object(valid_arr_filepath,valid_arr,'validation')

            logger.info(f"\n\n{'='*20}Data Transformation log completed.{'='*20} \n\n")          
            
        except Exception as e:
            raise e    
    
    