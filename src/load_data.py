import pandas as pd
from src.logger import logger
from src.exception_handle import DataLoadException
def data_loading(File_Path):
    try:

        data=pd.read_csv(File_Path)
        logger.info("Data has been loaded and converted to Datframe successfully")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise DataLoadException(e)
