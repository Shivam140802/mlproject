from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging      
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self,numerical_features, categorical_features):
        logging.info("Data Transformation started")
        try:
            logging.info("Train and Test datasets loaded successfully")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
      
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessor object created successfully")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Identifying features")
            numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_features = [col for col in numerical_features if col != 'Exam_Score']
            categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

            logging.info("Creating preprocessor object")
            preprocessor = self.get_data_transformation_object(numerical_features, categorical_features)

            target_column = 'Exam_Score'  
            
            input_features_train = train_df.drop(columns=[target_column],axis=1)
            target_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column],axis=1)
            target_test = test_df[target_column]

            logging.info("Applying fit_transform on train and transform on test")
            input_features_train_transformed = preprocessor.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor.transform(input_features_test)

            logging.info("saved preprocesssing object")
            save_object(
                file_path=self.transformation_config.preprocessor_path,
                obj=preprocessor
            )
            train_arr=np.c_[input_features_train_transformed,np.array(target_train)]
            test_arr=np.c_[input_features_test_transformed,np.array(target_test)]

            return (train_arr,
                    test_arr,
                    self.transformation_config.preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys) from e
