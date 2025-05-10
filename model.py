import os
import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from catboost import CatBoostRegressor

#Set paths for importing data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_data_folder = os.path.join(project_root, 'data')

print("Loading data...")

#Load data
train = pd.read_csv(os.path.join(path_to_data_folder, 'train.csv'))
test = pd.read_csv(os.path.join(path_to_data_folder, 'test.csv'))

print(f"Data loaded.")

#PREPROCESSING

#Remove id column
train = train.drop('id', axis=1) 
test_ids = test['id'] 
test = test.drop('id', axis=1) 

# Split features and target
X_temp = train.drop(columns='FloodProbability')
X = pd.DataFrame(index=train.index)
#X = X_temp

X['Siltation'] = train['Siltation']
X['MonsoonIntensity'] = train['MonsoonIntensity']
"""
X['IneffectiveDisasterPreparedness'] = train['IneffectiveDisasterPreparedness']
X['InadequatePlanning'] = train['InadequatePlanning']
X['DamsQuality'] = train['DamsQuality']
"""

y = train['FloodProbability']

X_temp_test = test
X_test = pd.DataFrame(index=test.index)

X_test['Siltation'] = test['Siltation']
X_test['MonsoonIntensity'] = test['MonsoonIntensity']


#No missing values
#Categorical variables already encoded on a scale of 1-10 already
#Order here carries meaning, so not one-hot encoding these

print("feature engineering")
#Feature engineering
X['sum_of_features'] = X_temp.sum(axis=1)
X['row_mean'] = X_temp.mean(axis=1)
X['row_median'] = X_temp.median(axis=1)
X['row_std'] = X_temp.std(axis=1)
X['row_q25'] = X_temp.quantile(0.25, axis=1)
X['row_q75'] = X_temp.quantile(0.75, axis=1)
X['row_max'] = X_temp.max(axis=1)
X['row_min'] = X_temp.min(axis=1) 

X_test['sum_of_features'] = X_temp_test.sum(axis=1)
X_test['row_mean'] = X_temp_test.mean(axis=1)
X_test['row_median'] = X_temp_test.median(axis=1)
X_test['row_std'] = X_temp_test.std(axis=1)
X_test['row_q25'] = X_temp_test.quantile(0.25, axis=1)
X_test['row_q75'] = X_temp_test.quantile(0.75, axis=1)
X_test['row_max'] = X_temp_test.max(axis=1)
X_test['row_min'] = X_temp_test.min(axis=1)


#MODEL 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
scores = []

for train_idx, val_idx in kf.split(X):
    print(f"Training fold {fold}...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    subsample=0.8,
    reg_lambda=2,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    print(f"Fold {fold} R²: {r2:.4f}")
    scores.append(r2)
    fold += 1

print(f"\nAverage R² across folds: {np.mean(scores):.4f}")

print("Training final model on full data...")
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    subsample=0.8,
    reg_lambda=2,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)
model.fit(X, y)

# Predict on test set
final_preds = model.predict(X_test)

#SUBMIT
submission = pd.DataFrame({
    'id': test_ids,
    'FloodProbability': final_preds
})
submission.to_csv('../submission/submission.csv', index=False)


print("done")