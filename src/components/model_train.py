import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor    

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainingConfig()
    
    def initialize_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and testing data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                'RandomForest':RandomForestRegressor(),
                'DecisionTree':DecisionTreeRegressor(),
                'Gradient boosting':GradientBoostingRegressor(),
                'linear Regressor':LinearRegression(),
                'KNN':KNeighborsRegressor(),
                'xgb':XGBRegressor(verbosity=0),
                'catboost':CatBoostRegressor(verbose=0),
                'adaboost':AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            best_model_name = max(model_report, key=lambda k: model_report[k]['r2_score'])
            best_model_score = model_report[best_model_name]['r2_score']
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )
            predictions=best_model.predict(X_test)
            final_r2_score=r2_score(y_test,predictions)
            logging.info(f"R2 score of the best model: {r2_score}")
            return final_r2_score
        except Exception as e:
            raise CustomException(e, sys) from e