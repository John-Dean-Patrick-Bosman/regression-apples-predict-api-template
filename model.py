"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    #model_train = train.loc[train['Commodities'] == 'APPLE GOLDEN DELICIOUS']
    #model_test = test.loc[test['Commodities'] == 'APPLE GOLDEN DELICIOUS']
    #feature_vector_df = train.loc[train['Commodities'] == 'APPLE GOLDEN DELICIOUS']
    #predict_structure = feature_vector_df[['avg_price_per_kg','Weight_Kg']]
    model_test = feature_vector_df.loc[feature_vector_df['Commodities'] == 'APPLE GOLDEN DELICIOUS']

    model_test = model_test.drop('Commodities',axis=1)

    #model_test = model_test.drop('Province',axis=1)
    model_test =  pd.get_dummies(model_test, drop_first = True, columns=['Province'])
    #model_test = model_test.drop('Size_Grade',axis=1)
    model_test =  pd.get_dummies(model_test, drop_first = True, columns=['Size_Grade'])
    #model_test = model_test.drop('Container',axis=1)
    model_test =  pd.get_dummies(model_test, drop_first = True, columns=['Container'])

    model_test['Month'] = pd.DatetimeIndex(model_test['Date']).month
    model_test = model_test.drop('Date',axis=1)

    model_test["Harvest_Season"] = model_test["Month"].astype(str)
    model_test["Harvest_Season"].replace({"1": "No", "2": "Yes", "12" : "No", "3" : "Yes", "4" : "No", "5" : "No", "6" : "No", "7": "No", "8" : "No", "9" : "No", "10" : "No", "11" : "No"}, inplace=True)
    model_test =  pd.get_dummies(model_test, drop_first = True, columns=['Harvest_Season'])
    model_test = model_test.drop('Index',axis=1)
    
    model_test.head()
                                
    # ------------------------------------------------------------------------

    return model_test

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
