from sklearn.metrics import accuracy_score, classification_report
import joblib


model = joblib.load("./models/random_forest.pkl")
X_test, y_test = joblib.load("./models/test_split.pkl")

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


