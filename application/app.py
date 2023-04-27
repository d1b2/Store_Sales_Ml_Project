from flask import Flask, render_template, request, session
import os
from Store_Sales.utils import *
from Store_Sales.constants import *
from Store_Sales import logger
from flask import Flask,request,app,jsonify,render_template,redirect,flash
from flask_sqlalchemy import SQLAlchemy
# Import for Migrations
from flask_migrate import Migrate, migrate
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from app_utils import *


secrets=read_yaml(SECRETS_FILE_PATH)
model = pickle.load(open('application/model.pkl', 'rb'))
transformer=pickle.load(open('application/model.pkl', 'rb'))


app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = secrets.FLASK_APP_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask_app.db' 
 
# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

class Profile(db.Model):
        id = db.Column(db.Integer, primary_key=True)       
        Item_Fat_Content = db.Column(db.String(20), unique=False, nullable=False)
        Outlet_Size = db.Column(db.String(20), unique=False, nullable=False)
        Item_Weight =  db.Column(db.Float, unique=False, nullable=False)
        Item_Category= db.Column(db.String(20), unique=False, nullable=False)
        Item_Type= db.Column(db.String(20), unique=False, nullable=False)
        Outlet_Identifier= db.Column(db.String(20), unique=False, nullable=False)
        Outlet_Location_Type= db.Column(db.String(20), unique=False, nullable=False)
        Outlet_Type=db.Column(db.String(20), unique=False, nullable=False)
        Item_MRP = db.Column(db.Float, unique=False, nullable=False)
        Item_Visibility = db.Column(db.Float, unique=False, nullable=False)       
        Prediction = db.Column(db.Float, unique=False, nullable=False)
        time_stamp=db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow())
        model_response_time=db.Column(db.Float, unique=False, nullable=False)      
    
        # repr method represents how one object of this datatable will look like
        def __repr__(self):
            return f"[{self.id},{self.Item_Fat_Content},{self.Outlet_Size},{self.Item_Weight},{self.Item_Category},\
                {self.Item_Type},{self.Outlet_Identifier},{self.Outlet_Location_Type},{self.Outlet_Type},\
                {self.Item_MRP},{self.Item_Visibility},{self.Prediction},{self.time_stamp},{self.model_response_time}]"

 
# Settings for migrations
migrate = Migrate(app, db)

#app routes
@app.route('/')
@app.route('/home')
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
    result=Profile.query.order_by(Profile.id.desc()).first().id
    data_to_show = Profile.query.get(result)
    return render_template('last_predict.html',values=data_to_show)

@app.route('/record')
def records():
    data = Profile.query.all()
    return render_template('records.html',data=data)

@app.route('/predict',methods=['GET','POST'])
def predict():
    test_df=pd.read_csv(CLEAN_TEST_FILE_PATH)
    item_type=list(test_df['Item_Type'].unique())
    outlet_identifier=list(test_df['Outlet_Identifier'].unique())
    if request.method=="POST":                  
            if request.form:                
                inputs=get_user_input_dataframe()
                start = datetime.datetime.utcnow()
                prediction = get_prediction(inputs)
                end = datetime.datetime.utcnow()
                response_time = round(((end-start).microseconds)/1e6,2)
                p=create_database_object(inputs.values.tolist()[0],prediction,response_time)               
                db.session.add(p)
                db.session.commit()
                print(prediction)
                flash(f" !! Prediction Done. Sale : {prediction} !! ")
                return render_template('predict.html',item_type=item_type,
                                       outlet_identifier=outlet_identifier)      
       
    else:
        print('Method not post')
        return render_template('predict.html',item_type=item_type,
                                    outlet_identifier=outlet_identifier)
    
@app.route('/delete/<int:id>')
def delete(id):
    # Deletes the data on the basis of unique id and
    # redirects to home page
    entry_to_delete = Profile.query.get_or_404(id)
    db.session.delete(entry_to_delete)
    db.session.commit()
    flash(" !! Record Deleted !! ")   
    return redirect('/record')

# json single record
@app.route('/record/<int:id>')
def recordJSON(id):
    data=Profile.query.filter((Profile.id == id)).first()  
    result= get_serialized_json(data)
    #print(dict1) 
    return (result)

# json all records
@app.route('/record/all')
def alljson():
     data = Profile.query.all()     
     results=[]
     for i in range(len(data)):
          results.append(get_serialized_json(data[i]))
     print(results)
     return jsonify(results)
     

if __name__=='__main__':
    app.run(debug = True)

