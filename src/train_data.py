import pandas as pd
import numpy as np
from src.logger import logger
from src.exception_handle import ModelTrainingException
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from src.config import RANDOM_STATE,TEST_SIZE,models,param_grids
from sklearn.model_selection import GridSearchCV

def train_test_data_split(X_data,Y_data):
    try:

        X_train,X_test,y_train,y_test=train_test_split(X_data,Y_data,random_state=RANDOM_STATE,test_size=TEST_SIZE)
        logger.info("Train Test Split has succesfully executed")
        return X_train,X_test,y_train,y_test
    
    except Exception as e:
        logger(f"Failed train test split {e}")
        raise ModelTrainingException(e)

def train_model(X_train_scaled,X_test_scaled,y_train,y_test):
    try:
        logger.info("model training has bee started")
        cv_results = {}
        for name, model in models.items():

            scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_results[name] = scores
            logger.info(f"{name} - CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

            # Create a dictionary of model names and their mean accuracy
            mean_scores = {name: scores.mean() for name, scores in cv_results.items()}

# Get the model with the highest average CV accuracy
            best_model_name = max(mean_scores, key=mean_scores.get)
            best_score = mean_scores[best_model_name]

            logger.info(f"\n✅ Best Model based on CV: {best_model_name} with Accuracy = {best_score:.4f}")

            return best_model_name,best_score
    except Exception as e:
        logger.error("Cross validation failed")  
        raise ModelTrainingException(e)



