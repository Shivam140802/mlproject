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

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple models and return their performance metrics."""
    
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            report[model_name] = {
                'r2_score': r2,
            }
        
        return report

    except Exception as e:
        raise CustomException(e, sys) from e
