import pandas as pd
import numpy as np
from src.logger import logger
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from src.exception_handle import PreprocessingException
from src.config import PROCESSED_FILE_PATH

def data_preproccessing(data):
    try:

        data["Health_Issues"]=data["Health_Issues"].fillna("Unknow")
        le=LabelEncoder()
        data["Stress_Lelevl_Lable"]=le.fit_transform(data["Stress_Level"])
        data=data.drop(columns="Stress_Level",axis=1)
        categorical_cols = ['Gender', 'Country', 'Sleep_Quality', 'Health_Issues', 'Occupation']
        df_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        bool_cols = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
        df_encoded.to_csv(PROCESSED_FILE_PATH)
        X_data=df_encoded.drop(columns=["ID","Stress_Lelevl_Lable"],axis=1)
        Y_data=df_encoded[["Stress_Lelevl_Lable"]]
        logger.info(f"Data has been proprocessed succesfully")
        return X_data,Y_data
    
    except Exception as e:
        logger.error(f"Error in preprocessing{e}")
        raise PreprocessingException(e)
 