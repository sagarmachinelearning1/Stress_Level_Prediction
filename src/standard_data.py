from sklearn.preprocessing import StandardScaler
from src.logger import logger
from src.exception_handle import StandardizationException
import joblib
from src.config import SCALED_MODEL

def data_standardization(X_train,X_test):
    try:
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        logger.info("Standardization has been succesfully done")
        joblib.dump(scaler,SCALED_MODEL)
        return X_train_scaled,X_test_scaled
    
    except Exception as e:
        logger.error(f"Standardization has been failed{e}")
        raise(e)

