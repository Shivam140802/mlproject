import dill
import os
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """Save any Python object to the specified path using joblib."""

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e,sys) from e 

def evaluate_model(X_train, y_train, X_test, y_test, model):
    """Evaluate multiple models and return their performance metrics."""
    
    try:
        model.fit(X_train, y_train)
        y_pred =model.predict(X_test)
        return r2_score(y_test, y_pred)
    
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path):
    """Load a Python object from the specified path using joblib."""
    
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys) from e
