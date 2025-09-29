# === Random Forest Classifier for Attrition Prediction ===
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

df = pd.read_csv("Employee_Performance_Retention.csv")

employee_ids = df["Employee_ID"] if "Employee_ID" in df.columns else None

if "Employee_ID" in df.columns:
    df = df.drop(columns=["Employee_ID"])

df = df.dropna()

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Attrition_Yes"])
y = df["Attrition_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

importances = rf.feature_importances_
feat_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(8,6))
plt.barh(feat_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

cleaned_df = df.copy()
if employee_ids is not None:
    cleaned_df.insert(0, "Employee_ID", employee_ids)

cleaned_df.to_csv("Employee_Performance_Retention_Cleaned.csv", index=False)
