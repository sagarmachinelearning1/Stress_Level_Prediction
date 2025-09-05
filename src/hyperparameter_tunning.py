import pandas as pd
import numpy as np
from src.logger import logger
from src.exception_handle import ModelTrainingException,HyperparametertunningException
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from src.config import RANDOM_STATE,TEST_SIZE,models,param_grids,MODEL_SAVED_PATH,REPORTS_SAVED_PATH
from sklearn.model_selection import GridSearchCV
import joblib


def hyper_parameter_tunning(best_model_name,X_train_scaled,X_test_scaled,y_train,y_test):
    try:

        for index,value in param_grids.items():

            if index==best_model_name:
                grid = GridSearchCV(models[index], param_grids[index], cv=5, scoring='accuracy')
                grid.fit(X_train_scaled, y_train) 
                logger.info(f"Best Parameters: {grid.best_params_}")
                logger.info(f"Best CV Score: {grid.best_score_:.4f}")


        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test_scaled)

        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        
        with open(REPORTS_SAVED_PATH, "w") as file:

         
            report = (
            f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
            f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}\n"
            f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
            file.write(report)

        joblib.dump(best_model,MODEL_SAVED_PATH)
        logger.info("Best model has been saved")
        return "Hyperparameter tunning is done and model saved"

    except Exception as e:
        logger.error("Hyperparameter tunning failed{e}")     
        raise HyperparametertunningException(e)


