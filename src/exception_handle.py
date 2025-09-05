# src/exceptions.py

class DataLoadException(Exception):
    """Raised when there is an error while loading the data."""
    def __init__(self, message="Failed to load data."):
        super().__init__(message)


class ModelTrainingException(Exception):
    """Raised when an error occurs during model training."""
    def __init__(self, message="Model training failed."):
        super().__init__(message)


class PredictionException(Exception):
    """Raised when an error occurs during prediction or inference."""
    def __init__(self, message="Prediction failed."):
        super().__init__(message)

class PreprocessingException(Exception):
    """Raised when an error occurs during preprocessing."""
    def __init__(self, message="Preprocessing failed."):
        super().__init__(message)        

class StandardizationException(Exception):
    """Raised when an error occurs during Standardization."""
    def __init__(self, message="Standardization failed."):
        super().__init__(message)   

class HyperparametertunningException(Exception):
    """Raised when an error occurs during Hyperparameter tunning."""
    def __init__(self, message="Hyperparameter tunning failed."):
        super().__init__(message)           

class PredictionException(Exception):
    """Raised when an error occurs during Prediction."""
    def __init__(self, message="Prediction failed."):
        super().__init__(message)                   