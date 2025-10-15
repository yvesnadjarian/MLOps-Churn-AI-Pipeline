import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Loading Data and One-hot Encoding
dt = pd.read_csv("./data/raw/telco.csv")
encoder = OneHotEncoder(sparse_output=False)

dt_encoded = dt.drop(columns='customerID')
categorial_cols = ['gender', 'Partner', 'Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod','Churn']

encoded_array = encoder.fit_transform(dt_encoded[categorial_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorial_cols))

final_dt = pd.concat([dt_encoded.drop(columns=categorial_cols), encoded_df], axis=1)

#Checking for Strings if needed (DEBUG)
#print(X_train.select_dtypes(include='object').columns)

# Reeplace Empty Values with NA
final_dt.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
final_dt.fillna(0, inplace=True)

final_dt.to_csv("./data/processed/telco_processed.csv", encoding='utf-8', index=False, header=True)
