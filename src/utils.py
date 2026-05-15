import os
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Serialize and save any Python object (model, preprocessor) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a serialized Python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Train each model with GridSearchCV hyperparameter tuning.
    Returns a dict of {model_name: f1_score_on_test}.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            # GridSearchCV with F1 as the scoring metric (correct for imbalanced data)
            gs = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)

            # Use best estimator found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]
                                if hasattr(model, "predict_proba") else y_pred)

            report[model_name] = {"f1": round(f1, 4), "roc_auc": round(auc, 4)}
            logging.info(f"{model_name} — F1: {f1:.4f}, AUC: {auc:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)