from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import f1_score
import xgboost as xgb
import json
import pandas as pd
df = pd.read_csv("dataset/processed/train_encoded.csv")
df["damage_grade"] = df["damage_grade"].astype('int')
df["damage_grade"] = df["damage_grade"] - 1

X = df.drop(["damage_grade"], axis=1)
y = df["damage_grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69420)


def xgb_objective(trial):
    params = {
        'tree_method':'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': 10000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }

    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    return f1_score(y_test, y_pred, average='macro')


study = optuna.create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=10)

print(study.best_params)

with open("metrics/xgb_tuning.json", "w") as f:
    json.dump(study.best_params, f)
