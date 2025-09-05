import os
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
RAW_FILE_PATH=os.path.join(BASE_DIR,"data","raw","Stress_Level_Prediction.csv")
PROCESSED_FILE_PATH=os.path.join(BASE_DIR,"data","processed","processed.csv")
MODEL_SAVED_PATH=os.path.join(BASE_DIR,"models","best_model.pkl")
SCALED_MODEL=os.path.join(BASE_DIR,"models","scaled.pkl")
LOGGING_PATH=os.path.join(BASE_DIR,"logs","record.log")
REPORTS_SAVED_PATH=os.path.join(BASE_DIR,"reports","reports.txt")


RANDOM_STATE=42
TEST_SIZE=0.30

models={
    "Logistic_Regression":LogisticRegression(),
    "Xgboost":XGBClassifier(),
    "Decision_Tree":DecisionTreeClassifier(),
    "Random_Forest":RandomForestClassifier(),
    "Gradient_Boosting":GradientBoostingClassifier(),
    "SVC":SVC()

}

param_grids = {
    "Logistic_Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    },

    "Xgboost": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.7, 1]
    },

    "Decision_Tree": {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    },

    "Random_Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'bootstrap': [True, False]
    },

    "Gradient_Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1.0]
    },

    "SVC": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}
