# === Support Vector Machine (SVM) for Attrition Prediction ===
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Load dataset
df = pd.read_csv("Employee_Performance_Retention.csv")

# Keep Employee_ID for saving cleaned dataset later
employee_ids = df["Employee_ID"] if "Employee_ID" in df.columns else None

# 2. Drop ID column from features
if "Employee_ID" in df.columns:
    df = df.drop(columns=["Employee_ID"])

# 3. Handle missing values
df = df.dropna()

# 4. Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 5. Define features and target
X = df.drop(columns=["Attrition_Yes"])
y = df["Attrition_Yes"]

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Try different kernels
kernels = ["linear", "poly", "rbf"]
results = {}

for kernel in kernels:
    print(f"\n=== Training SVM with {kernel} kernel ===")
    svm = SVC(kernel=kernel, random_state=42, class_weight="balanced")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[kernel] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Print metrics
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("\nDetailed report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"SVM Confusion Matrix ({kernel} kernel)")
    plt.show()

# 8. Compare kernels
results_df = pd.DataFrame(results).T
print("\n=== Kernel Comparison ===\n")
print(results_df)

# 9. Save cleaned dataset (restore Employee_ID if available)
cleaned_df = df.copy()
if employee_ids is not None:
    cleaned_df.insert(0, "Employee_ID", employee_ids)

cleaned_df.to_csv("Employee_Performance_Retention_Cleaned.csv", index=False)
