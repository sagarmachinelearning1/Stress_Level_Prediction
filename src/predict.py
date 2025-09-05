from src.config import MODEL_SAVED_PATH
from src.config import SCALED_MODEL
import joblib
from src.logger import logger
from src.exception_handle import PredictionException
def prediction(input_values):
    try:
        logger.info("User data predicttion has started")
        scaled=joblib.load(SCALED_MODEL)
        model=joblib.load(MODEL_SAVED_PATH)
        scaled_data=scaled.transform(input_values)
        predicted_value=model.predict(scaled_data)
        logger.info("Prediction has been done")
        return predicted_value
    
    except Exception as e:
        logger.error(f"prediction failed{e}")
        raise PredictionException(e)





