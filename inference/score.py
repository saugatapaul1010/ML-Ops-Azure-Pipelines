import os
import sys
import numpy as np
import joblib
#from sklearn.externals import joblib

import math
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import json
import re
import traceback
import logging
from sklearn.tree import DecisionTreeClassifier

'''
Inference script for AUTOANY Classification:

'''

#Things to pass from UI
#AUTOANY - registered model name
#message -list of independent variables
#model_path = autoany_model.pkl
#Action_taken_to_solve - target columns

def init():
    '''
    Initialize required models:
        Get the AUTOANY Model from Model Registry and load
    '''
    global prediction_dc
    global model
    prediction_dc = ModelDataCollector("AUTOANY", designation="predictions", feature_names=["message"])

    model_path = Model.get_model_path('AUTOANY')
    model = joblib.load(model_path+"/"+"autoany_model.pkl")
    print('AUTOANY model loaded...')

def create_response(predicted_lbl):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted AUTOANY Species
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    print("Predicted Action_taken_to_solve : ",predicted_lbl)
    resp_dict["predicted_Action_taken_to_solve"] = str(predicted_lbl)
    return json.loads(json.dumps({"output" : resp_dict}))

def run(raw_data):
    '''
    Get the inputs and predict the AUTOANY Species
    Arguments : 
        raw_data : message
    Returns :
        Predicted AUTOANY Species
    '''
    try:
        data = json.loads(raw_data)
        message = data['message']
        predicted_Action_taken_to_solve = model.predict([[message]])[0]
        prediction_dc.collect([message,predicted_Action_taken_to_solve])
        return create_response(predicted_Action_taken_to_solve)
    except Exception as err:
        traceback.print_exc()