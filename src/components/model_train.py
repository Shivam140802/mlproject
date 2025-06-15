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
                'KNN':KNeighborsRegressor(),
                'xgb':XGBRegressor(verbosity=0),
                'catboost':CatBoostRegressor(verbose=0),
                'adaboost':AdaBoostRegressor()
            }
            params={
                'randomforest':{
                    'n_estimators':[100, 200, 300],
                    'max_depth':[10, 20, 30 ,40],
                    'min_samples_split':[2, 5, 10],
                    'min_samples_leaf':[1, 2, 4],
                },
                'decisiontree':{
                    'max_depth':[10, 20, 30],
                    'min_samples_split':[2, 5, 10],
                    'min_samples_leaf':[1, 2, 4]
                },  
                'knn':{
                    'n_neighbors':[3, 5, 7, 9],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'xgb':{
                    'n_estimators':[100, 200, 300],
                    'learning_rate':[0.01, 0.1, 0.2],
                    'max_depth':[3, 5, 7]
                },
                'catboost':{
                    'iterations':[100, 200, 300],
                    'learning_rate':[0.01, 0.1, 0.2],
                    'depth':[3, 5, 7, 8, 9],
                    'l2_leaf_reg':[1, 3, 5]
                },
                'adaboost':{
                    'n_estimators':[50, 100, 200],
                    'learning_rate':[0.01, 0.1, 0.2],
                    'estimator':[DecisionTreeRegressor(max_depth=3), LinearRegression()]
                }
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            best_model_name = max(model_report, key=lambda k: model_report[k]['r2_score'])
            best_model_score = model_report[best_model_name]['r2_score']
            best_model=model_report[best_model_name]['model']

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