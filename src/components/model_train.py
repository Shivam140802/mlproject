import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.metrics import r2_score
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
            model=AdaBoostRegressor(
                estimator=Ridge(alpha=0.23),   
                n_estimators=180,
                learning_rate=1
            )
            r2=evaluate_model(X_train,y_train,X_test,y_test,model)
            logging.info(f"R2 score of the model: {r2}")

            save_object(
                file_path=self.model_train_config.trained_model_file_path,
                obj=model
            )
            predictions=model.predict(X_test)
            final_r2_score=r2_score(y_test,predictions)
            logging.info(f"R2 score of the best model: {r2_score}")
            return final_r2_score
        except Exception as e:
            raise CustomException(e, sys) from e