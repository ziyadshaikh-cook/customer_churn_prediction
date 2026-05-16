import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test arrays into X, y")
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test  = test_arr[:, :-1]
            y_test  = test_arr[:, -1]

            # --- All models to evaluate ---
            models = {
                "Logistic Regression":  LogisticRegression(max_iter=1000),
                "Decision Tree":        DecisionTreeClassifier(),
                "Random Forest":        RandomForestClassifier(random_state=42),
                "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
                "XGBoost":              XGBClassifier(eval_metric="logloss", random_state=42),
                "CatBoost":             CatBoostClassifier(verbose=0, random_state=42,train_dir = "artifacts/catboost_info"),
                "SVM":                  SVC(probability=True),
            }

            # --- Hyperparameter grids for GridSearchCV ---
            # Only tune the top performers — tuning all 7 wastes time
            params = {
                "Logistic Regression":  {"C": [0.1, 1, 10]},
                "Decision Tree":        {"max_depth": [3, 5, 7]},
                "Random Forest":        {"n_estimators": [100, 200], "max_depth": [5, 10]},
                "Gradient Boosting":    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                "XGBoost":              {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "CatBoost":             {"iterations": [100, 200], "learning_rate": [0.05, 0.1]},
                "SVM":                  {"C": [0.1, 1], "kernel": ["rbf"]},
            }

            # --- Train, tune, score all models ---
            # evaluate_models is in utils.py — it runs GridSearchCV and returns F1 + AUC
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            logging.info(f"Model evaluation report: {model_report}")

            # --- Pick the best model by F1 score ---
            best_model_name = max(model_report, key=lambda k: model_report[k]["f1"])
            best_model_score = model_report[best_model_name]["f1"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with F1: {best_model_score}")

            # Minimum acceptable F1 — if nothing clears this, the pipeline fails loudly
            if best_model_score < 0.55:
                raise CustomException("No model achieved acceptable F1 score (>0.55)", sys)

            # --- Save the best model ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # --- Final score on test set ---
            y_pred = best_model.predict(X_test)
            final_f1  = f1_score(y_test, y_pred)
            final_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            logging.info(f"Final Test F1: {final_f1:.4f} | AUC: {final_auc:.4f}")

            return final_f1, final_auc, best_model_name

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_arr = np.load("artifacts/train_arr.npy")
    test_arr  = np.load("artifacts/test_arr.npy")

    trainer = ModelTrainer()
    f1, auc, best_model = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"\nBest Model : {best_model}")
    print(f"Final F1   : {f1:.4f}")
    print(f"Final AUC  : {auc:.4f}")