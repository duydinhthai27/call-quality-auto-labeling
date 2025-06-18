# --- config.py ---
CONFIDENCE_THRESHOLD = 0.9
LABELED_RATIO = 0.1
CHECKPOINT_DIR = "model_checkpoints"
SAMPLE_DIR = "sample_predictions"

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os

# Create directories if not exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

MODEL_REGISTRY = {
    #'random_forest': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42, 'class_weight' : 'balanced'}),
    'logistic_regression': (LogisticRegression, {'solver': 'liblinear', 'random_state': 42}),
    'gradient_boosting': (GradientBoostingClassifier, {'n_estimators': 100, 'random_state': 42}),
    'xgboost': (XGBClassifier, {'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42}),
    'svm': (SVC, {'probability': True, 'random_state': 42}),
    'knn': (KNeighborsClassifier, {}),
    'sgd': (SGDClassifier, {'loss': 'log_loss', 'random_state': 42}),
    'decision_tree': (DecisionTreeClassifier, {'random_state': 42}),
    'adaboost': (AdaBoostClassifier, {'random_state': 42}),
    'extra_trees': (ExtraTreesClassifier, {'random_state': 42})
}
