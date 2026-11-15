import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    shap_summary_path = os.path.join("artifacts", "shap_summary.png")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Split features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Proceeding without SMOTE. Using original class distribution.")

            models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50),
                "CatBoost": CatBoostClassifier(random_state=42, verbose=False, iterations=50),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
                "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=50),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5]
                },
                "XGBoost": {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.2]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                },
                "CatBoost": {
                    'iterations': [50, 100],
                    'depth': [4, 6],
                    'learning_rate': [0.1, 0.2]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                },
                
            }

            logging.info("Starting GridSearchCV for model selection")
            model_report, fitted_models = evaluate_models(X_train, y_train, X_test, y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = fitted_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No sufficiently good model found")

            logging.info(f"Best model found: {best_model_name} (ROC-AUC: {best_model_score:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            y_test_prob = best_model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predicted)
            roc_auc = roc_auc_score(y_test, y_test_prob)
            report = classification_report(y_test, predicted, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predicted)

            logging.info(f"Classification report:\n{classification_report(y_test, predicted)}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            print("\n" + "="*60)
            print(f" BEST MODEL: {best_model_name}")
            print(f" ROC-AUC: {roc_auc:.4f}")
            print(f" ACCURACY: {accuracy:.4f}")
            print(f" PRECISION: {report['macro avg']['precision']:.4f}")
            print(f" RECALL: {report['macro avg']['recall']:.4f}")
            print(f" F1-SCORE: {report['macro avg']['f1-score']:.4f}")
            print("="*60)

           
           

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)
