print("\n===== AdaBoost (WITH Imputation) =====")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from evaluation_utils import evaluate_model, print_results

df = pd.read_csv("final_clean_for_ml.csv")

if "SITE_ID_y" in df.columns:
    df = df.drop(columns=["SITE_ID_y"])

X = df.drop(columns=["DX_GROUP"]).values
y = df["DX_GROUP"].values

print("Missing values before:", np.isnan(X).sum())

# Median imputation
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

print("Missing values after:", np.isnan(X).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

stump = DecisionTreeClassifier(max_depth=1)

model = AdaBoostClassifier(
    estimator=stump,
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

results = evaluate_model(y_test, y_pred, y_proba)
print_results(results)
