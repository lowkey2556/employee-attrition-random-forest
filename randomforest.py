# === Random Forest Classifier for Attrition Prediction ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("Employee_Performance_Retention.csv")

if "Employee_ID" in df.columns:
    df = df.drop(columns=["Employee_ID"])

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop(columns=["Attrition"])   
y = df["Attrition"]                  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,   
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

importances = rf.feature_importances_
feat_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(8,6))
plt.barh(feat_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()