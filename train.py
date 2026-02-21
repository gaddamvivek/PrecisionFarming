import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

print("Loading training data...")
df = pd.read_csv("data/crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

model_defs = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss"),
    "SVM":                 SVC(kernel="rbf", random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Naive Bayes":         GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
}

FILE_NAMES = {
    "Random Forest":       "RandomForest",
    "XGBoost":             "XGBoost",
    "SVM":                 "SVMClassifier",
    "Decision Tree":       "DecisionTree",
    "Naive Bayes":         "NBClassifier",
    "Logistic Regression": "LogisticRegression",
}

accuracies = {}
feature_importances = {}

for name, model in model_defs.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
    accuracies[name] = acc
    print(f"  {name}: {acc}%")

    with open(f"models/{FILE_NAMES[name]}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Feature importances (RF and XGBoost only)
    if hasattr(model, "feature_importances_"):
        feature_importances[name] = {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURES, model.feature_importances_)
        }

# Crop distribution
crop_counts = df["label"].value_counts().to_dict()

metrics = {
    "accuracies": accuracies,
    "feature_importances": feature_importances,
    "crop_counts": crop_counts,
    "total_samples": len(df),
    "num_crops": len(crop_counts),
    "num_features": len(FEATURES),
    "features": FEATURES,
}

with open("data/model_metrics.json", "w") as f:
    json.dump(metrics, f)

print("All models trained and metrics saved.")
