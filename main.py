import pandas as pd
import numpy as np
from src.config import RAW_FILE_PATH,PROCESSED_FILE_PATH
from src.logger import logger
from src.exception_handle import DataLoadException,PredictionException,ModelTrainingException,PreprocessingException
from src.load_data import data_loading
from src.preprocess_data import data_preproccessing
from src.train_data import train_test_data_split,train_model
from src.standard_data import data_standardization
from src.hyperparameter_tunning import hyper_parameter_tunning

def main():

    logger.info("Data loaading has been started")
    data=data_loading(RAW_FILE_PATH)
    logger.info("Data loaded succesfully")
    
    logger.info("Data preprocessing has been started")
    X_data,Y_data=data_preproccessing(data)
    logger.info("Data preprocessing has done successfully")
    logger.info("Train test split has been started")
    X_train,X_test,y_train,y_test=train_test_data_split(X_data,Y_data)
    logger.info("Train test split done succefully")
    logger.info("Standardization started")
    X_train_scaled,X_test_scaled=data_standardization(X_train,X_test)
    logger.info("Standardization has been succesfully done")
    best_model_name,best_score=train_model(X_train_scaled,X_test_scaled,y_train,y_test)
    logger.info("Best model name{best_model_name}","Best score{best_score}")
    logger.info("Best model has beeb selected")
    logger.info("Hyperparameter tunning started")
    info=hyper_parameter_tunning(best_model_name,X_train_scaled,X_test_scaled,y_train,y_test)
    logger.info(info)



if __name__=="__main__":
    main()