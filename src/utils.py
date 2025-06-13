import joblib
import os
import sys
from src.exception import CustomException
import pandas as pd
import numpy as np



def save_object(file_path, obj):
    """Save any Python object to the specified path using joblib."""

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e,sys) from e
