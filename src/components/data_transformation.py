import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds and returns the ColumnTransformer pipeline.
        This is ONLY the preprocessor — SMOTE is applied AFTER this.
        """
        try:
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
            categorical_columns = [
                "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"
            ]

            # Impute median for numerical NaNs (the 11 blank TotalCharges rows)
            # then scale to mean=0, std=1
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # OneHotEncode all categorical columns
            # handle_unknown='ignore' prevents crashes on unseen categories at inference
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", numerical_pipeline, numerical_columns),
                ("cat_pipeline", categorical_pipeline, categorical_columns)
            ])

            logging.info("Preprocessor ColumnTransformer built successfully.")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            # --- Clean TotalCharges: convert string → float ---
            # Blank spaces become NaN, handled by SimpleImputer(median) in pipeline
            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")
            logging.info("TotalCharges converted to numeric.")

            # --- Drop customerID ---
            train_df.drop(columns=["customerID"], inplace=True)
            test_df.drop(columns=["customerID"], inplace=True)

            # --- Encode target: Yes → 1, No → 0 ---
            train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})
            test_df["Churn"] = test_df["Churn"].map({"Yes": 1, "No": 0})

            target_column = "Churn"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info(f"Train shape before SMOTE: X={X_train.shape}, y distribution: {y_train.value_counts().to_dict()}")

            # --- Fit preprocessor on train, transform both ---
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Preprocessing applied. Scaler fit on train only — no data leakage.")

            # --- Apply SMOTE on TRAIN only ---
            # Never on test. Test must reflect real-world distribution.
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
            logging.info(f"SMOTE applied. Train shape after resampling: {X_train_resampled.shape}")
            logging.info(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

            # --- Combine features + target into arrays ---
            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # --- Save preprocessor ---
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor saved to: {self.data_transformation_config.preprocessor_obj_file_path}")

            np.save(os.path.join("artifacts", "train_arr.npy"), train_arr)
            np.save(os.path.join("artifacts", "test_arr.npy"), test_arr)
            logging.info("Train and test arrays saved to artifacts/")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Run ingestion first to ensure train.csv and test.csv exist
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path, test_path
    )

    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")