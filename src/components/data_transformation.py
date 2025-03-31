import sys
import os
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass


@dataclass  # No need to manually create __init__
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]

            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data train and test loading completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_object = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = train_df.drop(target_column_name, axis=1)
            target_feature_test_df = train_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_array = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_object.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array, np.array(input_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(input_feature_test_df)]

            logging.info('Saving preprocessing object')

            save_object(file_path=DataTransformationConfig.preprocessor_obj_file_path, obj=preprocessor_object)

            return (
                train_array,
                test_array,
                DataTransformationConfig.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
