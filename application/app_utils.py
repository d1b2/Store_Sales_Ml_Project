import pandas as pd
from flask import  request
import os
import csv
import datetime
import pickle
from Store_Sales.constants import *
from app import Profile

def create_database_object(input_list,prediction,response_time):    
    p = Profile(       
        Item_Fat_Content = input_list[0],
        Outlet_Size =input_list[1],
        Item_Weight=input_list[2],
        Item_Category= input_list[3],
        Item_Visibility = input_list[4],
        Item_Type= input_list[5],
        Item_MRP = input_list[6],        
        Outlet_Identifier= input_list[7],
        Outlet_Location_Type= input_list[8],
        Outlet_Type= input_list[9],      
        Prediction = prediction ,      
        model_response_time=response_time
         )
    return p



def get_user_input_dataframe():
    """Gets user inputs and convert it into dataframe

    Returns:
        user_inputs(panadas dataframe): converted dataframe
    """
    try:
        Item_Fat_Content=request.form.get('Item_Fat_Content')
        Outlet_Size=request.form.get('Outlet_Size')
        Item_Weight=request.form.get('Item_Weight')
        Item_Category=request.form.get('Item_Category')
        Item_Visibility=request.form.get('Item_Visibility')
        Item_Type=request.form.get('Item_Type')
        Item_MRP=request.form.get('Item_MRP')
        Outlet_Identifier=request.form.get('Outlet_Identifier')
        Outlet_Location_Type=request.form.get('Outlet_Location_Type')
        Outlet_Type=request.form.get('Outlet_Type')

        user_inputs = pd.DataFrame([[Item_Fat_Content,Outlet_Size,Item_Weight,Item_Category,
                                     Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,
                                     Outlet_Location_Type,Outlet_Type]], 
                    columns=['Item_Fat_Content','Outlet_Size','Item_Weight','Item_Category',
                            'Item_Visibility','Item_Type','Item_MRP','Outlet_Identifier',
                            'Outlet_Location_Type','Outlet_Type'])
        return user_inputs
        
    except Exception as e:
        raise e

def get_prediction(dataframe):
    model = pickle.load(open('application/model.pkl', 'rb'))
    transformer=pickle.load(open('application/column_transformer.pkl', 'rb'))
    transformed_df=transformer.transform(dataframe)
    #print(transformed_df)
    predictions=model.predict(transformed_df)
    predictions=round(predictions[0], 2)
    #predictions=transformed_df
    print(predictions)
    return predictions
dataframe=pd.read_csv(CLEAN_TEST_FILE_PATH)
get_prediction(dataframe)

def get_serialized_json(element):
    json_dict=element.__dict__
    json_dict.pop('_sa_instance_state')
    json_dict = {k.upper():v for k,v in json_dict.items()}
   
    return json_dict


