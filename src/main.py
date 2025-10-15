import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


#Loading Data and One-hot Encoding
dt = pd.read_csv("./data/telco.csv")
encoder = OneHotEncoder(sparse_output=False)

dt_encoded = dt.drop(columns='customerID')
categorial_cols = ['gender', 'Partner', 'Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod','Churn']

encoded_array = encoder.fit_transform(dt_encoded[categorial_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorial_cols))

final_dt = pd.concat([dt_encoded.drop(columns=categorial_cols), encoded_df], axis=1)


#Checking for Strings
#print(X_train.select_dtypes(include='object').columns)


# Reeplace Empty Values with NA
final_dt.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
final_dt.fillna(0, inplace=True)


# Spliting data and training the model
X = final_dt.drop(columns=["Churn_No","Churn_Yes"])
y = final_dt["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Saving the model
joblib.dump(model, "./models/random_forest.pkl")