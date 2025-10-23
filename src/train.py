import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# loading Processed Data
processed_dt = pd.read_csv("./data/processed/telco_processed.csv")

# Spliting data and training the model
X = processed_dt.drop(columns=["Churn_No","Churn_Yes"])
y = processed_dt["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, "./models/random_forest.pkl")