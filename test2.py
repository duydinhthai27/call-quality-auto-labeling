import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Load data
file_path = './Filtered_Call_Records.csv'
df = pd.read_csv(file_path)

# Clean initial data
def clean_initial_data(df):
    # Convert mixed-type columns to string for inf values
    df = df.applymap(lambda x: str(x) if isinstance(x, (int, float)) and x in [np.inf, -np.inf] else x)
    # Drop irrelevant columns
    cols_to_drop = [
        'StreamId', 'CallRecordId', 'SegmentFailedReason', 'SegmentFailureStage',
        'CallerPhoneNumber', 'CalleePhoneNumber', 'Comment', 'CallStartTime',
        'CallEndTime', 'CallerIpAddress', 'SbcSessionId', 'SbcSessionStartTime',
        'SbcSessionEndTime', 'OverallCallQuality', 'CallerReflexiveIpAddress',
        'CallerRelayIpAddress', 'CallerRelayPort', 'CallMediaPathLocation'
    ]
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns])

df_clean = clean_initial_data(df)
df_clean = df_clean.dropna(subset=['StreamQuality'])

# Split data before any preprocessing
X = df_clean.drop(columns=['StreamQuality'])
y = df_clean['StreamQuality']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Preserve class distribution
)

# Encode target variable
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_test = target_encoder.transform(y_test)

# Custom transformer for safe numeric conversion
class SafeNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(pd.to_numeric, errors='coerce')

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('convert', SafeNumericTransformer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Modified categorical transformer to convert all values to string
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('to_str', FunctionTransformer(lambda X: X.astype(str))),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models with balanced class weights where applicable
models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=1),
    'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, silent=True),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Feature importance visualization function
def plot_feature_importance(importances, feature_names, model_name):
    top_n = 10 
    indices = np.argsort(importances)[-top_n:][::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances - {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

# Training and evaluation function
def train_and_evaluate(models, preprocessor, use_smote=False):
    results = []
    
    for name, model in models.items():
        if use_smote:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
        print(f"\n{'='*40}")
        print(f"Training {name} with{'out' if not use_smote else ''} SMOTE")
        print(f"{'='*40}")
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Get feature names after preprocessing
        try:
            feature_names = (pipeline.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .named_steps['onehot']
                             .get_feature_names_out(categorical_features))
            feature_names = np.concatenate([
                numeric_features,
                feature_names
            ])
        except Exception as e:
            feature_names = numeric_features + categorical_features
        
        # Get feature importances
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                raise AttributeError("No feature importance available")
            
            plot_feature_importance(importances, feature_names, name)
        except Exception as e:
            print(f"Could not get feature importances for {name}: {str(e)}")
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
        
        results.append({
            'model': name,
            'accuracy': accuracy,
            'report': report
        })
        
        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")
    
    return pd.DataFrame(results)

# Run evaluations
print("Training without SMOTE:")
normal_results = train_and_evaluate(models, preprocessor)

print("\nTraining with SMOTE:")
smote_results = train_and_evaluate(models, preprocessor, use_smote=True)

# Save results
normal_results.to_csv('normal_training_results.csv', index=False)
smote_results.to_csv('smote_training_results.csv', index=False)

# Print final summary
print("\nFinal Results Summary:")
print("Without SMOTE:")
print(normal_results[['model', 'accuracy']])
print("\nWith SMOTE:")
print(smote_results[['model', 'accuracy']])
