from flask import Flask, render_template, request, session
import os
from Store_Sales.utils import *
from Store_Sales.constants import *
from Store_Sales import logger
from flask import Flask,request,app,jsonify,url_for,render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from app_utils import *

secrets=read_yaml(SECRETS_FILE_PATH)
model = pickle.load(open('application/model.pkl', 'rb'))
transformer=pickle.load(open('application/model.pkl', 'rb'))


app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = secrets.FLASK_APP_KEY

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/last_predict')
def last_predict():
    return render_template('last_predict.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    test_df=pd.read_csv(CLEAN_TEST_FILE_PATH)
    item_type=list(test_df['Item_Type'].unique())
    outlet_identifier=list(test_df['Outlet_Identifier'].unique())
    if request.method=="POST":
       #try:            
            if request.form:
                inputs=get_user_input_dataframe()
                prediction = get_prediction(inputs)
                print(prediction)
                
                return render_template('predict.html',item_type=item_type,
                                       outlet_identifier=outlet_identifier)
       
       
    else:
        print('Method not post')
        return render_template('predict.html',item_type=item_type,
                                    outlet_identifier=outlet_identifier)

if __name__=='__main__':
    app.run(debug = True)

