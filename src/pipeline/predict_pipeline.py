import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging  
from src.utils import load_object


class PredictPipeline:
    def __init__(self,model=None, preprocessor=None):
        self.model =model or load_object('artifacts/model.pkl')
        self.preprocessor =preprocessor or load_object('artifacts/preprocessor.pkl')
    
    def predict(self, features):
        try:
            logging.info("Starting prediction")
            data_scaled = self.preprocessor.transform(features)
            logging.info("Data transformed using preprocessor")
            predictions = self.model.predict(data_scaled)
            logging.info("Prediction completed")
            return predictions
        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(self,Hours_Studied:int, Attendance:int, Parental_Involvement:str,
       Access_to_Resources:str, Extracurricular_Activities:str, Sleep_Hours:int,
       Previous_Scores:int, Motivation_Level:str, Internet_Access:str,
       Tutoring_Sessions:int, Family_Income:str, Teacher_Quality:str, School_Type:str,
       Peer_Influence:str, Physical_Activity:int, Learning_Disabilities:str,
       Parental_Education_Level:str, Distance_from_Home:str, Gender:str):
        self.Hours_Studied = Hours_Studied
        self.Attendance = Attendance    
        self.Parental_Involvement = Parental_Involvement
        self.Access_to_Resources = Access_to_Resources
        self.Extracurricular_Activities = Extracurricular_Activities
        self.Sleep_Hours = Sleep_Hours
        self.Previous_Scores = Previous_Scores
        self.Motivation_Level = Motivation_Level
        self.Internet_Access = Internet_Access
        self.Tutoring_Sessions = Tutoring_Sessions
        self.Family_Income = Family_Income
        self.Teacher_Quality = Teacher_Quality
        self.School_Type = School_Type
        self.Peer_Influence = Peer_Influence
        self.Physical_Activity = Physical_Activity
        self.Learning_Disabilities = Learning_Disabilities
        self.Parental_Education_Level = Parental_Education_Level
        self.Distance_from_Home = Distance_from_Home
        self.Gender=Gender
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Hours_Studied': [self.Hours_Studied],
                'Attendance': [self.Attendance],
                'Parental_Involvement': [self.Parental_Involvement],
                'Access_to_Resources': [self.Access_to_Resources],
                'Extracurricular_Activities': [self.Extracurricular_Activities],
                'Sleep_Hours': [self.Sleep_Hours],
                'Previous_Scores': [self.Previous_Scores],
                'Motivation_Level': [self.Motivation_Level],
                'Internet_Access': [self.Internet_Access],
                'Tutoring_Sessions': [self.Tutoring_Sessions],
                'Family_Income': [self.Family_Income],
                'Teacher_Quality': [self.Teacher_Quality],
                'School_Type': [self.School_Type],
                'Peer_Influence': [self.Peer_Influence],
                'Physical_Activity': [self.Physical_Activity],
                'Learning_Disabilities': [self.Learning_Disabilities],
                'Parental_Education_Level': [self.Parental_Education_Level],
                'Distance_from_Home': [self.Distance_from_Home],
                'Gender': [self.Gender]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e