import os
from Store_Sales.entity import *
from Store_Sales.utils import *
from Store_Sales.constants import *
from Store_Sales import logger
from sklearn.preprocessing import FunctionTransformer,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config
        self.train=pd.read_csv(TRAIN_FILE_PATH)
        self.test =pd.read_csv(TEST_FILE_PATH)
        
    def create_combined_X_df(self):
        try:           
            combined_X_df=pd.concat([self.train.iloc[:,:-1], self.test], ignore_index=True, axis=0)
            logger.info("Combined Dataframe created.")
            return combined_X_df
        except Exception as e:
            raise e
    
    def cleanfat(self,df):
        try:
            df['Item_Fat_Content'].replace({'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}, inplace=True)
            df.loc[df['Item_Identifier'].astype(str).str[:1] == 'N', 'Item_Fat_Content']='Not Applicable'
            df.drop(['Item_Identifier'],inplace=True,axis=1)
            logger.info("Item_Fat_Content column cleaned.")
            return df
        except Exception as e:
            raise e
        
    def custom_imputer_generator(self,df):
        try:
            ord_encoder=OrdinalEncoder()
            knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")   
            df1=df.iloc[:,0:2]
            df1['Item_Identifier']=ord_encoder.fit_transform(df1)
            df['Item_Weight']=knn_imputer.fit_transform(df1)[:,1]
            logger.info("Item_Weight column cleaned.")
            df['Item_Category']=df['Item_Identifier'].astype(str).str[:1].replace(['F', 'D', 'N'],['Food', 'Drink', 'Non_Consumable'])
            logger.info("Item_Category column created.")
            return df
        except Exception as e:
            raise e
    
    def get_preprocessor(self):
        try:
            get_clean_fat = FunctionTransformer(self.cleanfat)
            get_clean_weight = FunctionTransformer(self.custom_imputer_generator)
            impute_constant = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value='Small')

            preprocessor = make_column_transformer(
                (get_clean_fat, ['Item_Fat_Content','Item_Identifier']),              
                (impute_constant,['Outlet_Size']),
                (get_clean_weight,['Item_Identifier','Item_Weight']), 
                remainder='passthrough'
                )
            logger.info("Preprocessor called.")
            return preprocessor    
            
        except Exception as e:
            raise e


    def clean_combined_X_df(self):
        try:
            combined_X_df=self.create_combined_X_df()
            preprocessor=self.get_preprocessor()
            combined_cleaned_array=preprocessor.fit_transform(combined_X_df)
            logger.info("Preprocessor applied to Combined Dataframe.")
            columns_modified=['Item_Fat_Content','Outlet_Size','Item_Identifier','Item_Weight']
            columns_modified1=columns_modified+['Item_Category']
            clean_combined_X_df=pd.DataFrame(combined_cleaned_array,columns=columns_modified1+list(combined_X_df.columns.drop(columns_modified)))
            #clean_combined_X_df=pd.DataFrame(combined_cleaned_array)
            clean_combined_X_df=clean_combined_X_df[list(combined_X_df[:2])+['Item_Category']+list(combined_X_df[2:])]

            logger.info("Combined Dataframe is cleaned.")
            #return combined_X_df
            return clean_combined_X_df
            
        except Exception as e:
            raise e
        
    def train_validation_test_split(self):
        try:
            cleaned_dataframe=self.clean_combined_X_df()
            test_set=cleaned_dataframe.iloc[8523:,:]
            train_set=cleaned_dataframe.iloc[:8523,:]
            logger.info("Combined Dataframe splitted into cleaned test and cleaned train dataframes.")
            train_set['Item_Outlet_Sales']=self.train['Item_Outlet_Sales']
            logger.info("Target column added to cleaned train dataframe.")
            train_set['Sales_cat']=pd.cut(
                train_set["Item_Outlet_Sales"],
                bins=[0, 3000, 6000, 9000, 13100],
                labels=[1,2,3,4]
                )
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index,validation_index in split.split(train_set,train_set['Sales_cat']):
                strat_train_set = train_set.loc[train_index].drop(['Sales_cat'],axis=1)
                strat_validation_set= train_set.loc[validation_index].drop(['Sales_cat'],axis=1)
            logger.info("Cleaned train dataframe splitted into cleaned train and cleaned validation datasets.")
            cleaned_train_path=os.path.join(self.config.clean_csv_dir,self.config.clean_train_filename)
            cleaned_test_path=os.path.join(self.config.clean_csv_dir,self.config.clean_test_filename)
            cleaned_valid_path=os.path.join(self.config.clean_csv_dir,self.config.clean_validation_filename)
            strat_train_set.to_csv(cleaned_train_path,index=False)
            logger.info("Cleaned train dataset saved.")
            test_set.to_csv(cleaned_test_path,index=False)
            logger.info("Cleaned test dataframe saved.")
            strat_validation_set.to_csv(cleaned_valid_path,index=False)
            logger.info("Cleaned validation dataset saved.")

        except Exception as e:
            raise e

    def initiate_data_cleaning(self) :
        try:
            logger.info(f"\n\n{'='*20}Data Cleaning log started.{'='*20} \n\n")
            self.train_validation_test_split()        
            logger.info(f"\n\n{'='*20}Data Cleaning log completed.{'='*20} \n\n")          
            
        except Exception as e:
            raise e