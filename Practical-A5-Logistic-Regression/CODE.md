# Practical A5 — Data Analytics-II (Logistic Regression)

**Aim:** Implement Logistic Regression to classify whether a user purchased a product (`Purchased` = 0/1) using the Social Network Ads dataset, then compute the confusion matrix, TP, FP, TN, FN, accuracy, error rate, precision and recall.

**Dataset:** Social Network Ads — 400 rows.
**Source:** <https://www.kaggle.com/datasets/akram24/social-network-ads>
**File used:** `Assignment-A5-Social_Network_Ads.csv`

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Full Code

```python
# ============================================================
# Practical A5 : Logistic Regression (classification)
# Dataset : Social_Network_Ads.csv
# Target  : Purchased (0 = No, 1 = Yes)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ---------------- 1. Load the dataset ----------------
df = pd.read_csv("Assignment-A5-Social_Network_Ads.csv")
print("Shape:", df.shape)          # (400, 5)
print("\nFirst 5 rows:\n", df.head())
print("\nColumns:", df.columns)

# ---------------- 2. Data preprocessing ----------------
print("\nMissing values:\n", df.isnull().sum())

# 'User ID' is an identifier, not a predictor -> drop it
df = df.drop("User ID", axis=1)

# Encode the categorical column Gender -> 0/1
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

print("\nAfter encoding:\n", df.head())
print("\nClass distribution:\n", df["Purchased"].value_counts())

# ---------------- 3. Features and target ----------------
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]

# ---------------- 4. Train-test split (75 % / 25 %) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

# ---------------- 5. Feature scaling ----------------
# Age is ~18-60 while EstimatedSalary is ~15k-150k -> scaling is essential
sc = StandardScaler()
X_train = sc.fit_transform(X_train)   # fit ONLY on training data
X_test = sc.transform(X_test)         # reuse the same mean/std

# ---------------- 6. Train the Logistic Regression model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# ---------------- 7. Predictions ----------------
y_pred = model.predict(X_test)                    # 0 or 1
y_prob = model.predict_proba(X_test)[:, 1]        # probability of class 1

# ---------------- 8. Confusion matrix ----------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

TN, FP, FN, TP = cm.ravel()
print(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")

# ---------------- 9. Evaluation metrics ----------------
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = TN / (TN + FP)

print("\n----- Evaluation -----")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Error Rate  : {error_rate:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- 10. Visualise the confusion matrix ----------------
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Purchased", "Purchased"],
            yticklabels=["Not Purchased", "Purchased"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------- 11. Visualise the sigmoid (why output is a probability) ----------------
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid)
plt.axhline(0.5, color="red", linestyle="--")
plt.axvline(0, color="grey", linestyle="--")
plt.xlabel("z = b0 + b1x1 + ... + bnxn")
plt.ylabel("P(y = 1)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.show()

# ---------------- 12. Predict for a new user ----------------
new_user = pd.DataFrame({"Gender": [1], "Age": [35], "EstimatedSalary": [60000]})
new_user_scaled = sc.transform(new_user)
print("\nNew user -> class:", model.predict(new_user_scaled)[0],
      " probability of purchase:", round(model.predict_proba(new_user_scaled)[0][1], 4))
```

## Output

```text
Shape: (400, 5)

After encoding:
    Gender  Age  EstimatedSalary  Purchased
0        0   19            19000          0
1        0   35            20000          0
2        1   26            43000          0
3        1   27            57000          0
4        0   19            76000          0

Class distribution:
 Purchased
0    257
1    143

Train size: (300, 3)  Test size: (100, 3)

Intercept: [-1.138223]
Coefficients: [[-0.16838895  2.16551472  1.2163576 ]]
   (Gender, Age, EstimatedSalary)

Confusion Matrix:
 [[59  5]
 [14 22]]
TP = 22, FP = 5, TN = 59, FN = 14

----- Evaluation -----
Accuracy    : 0.8100
Error Rate  : 0.1900
Precision   : 0.8148
Recall      : 0.6111
F1-Score    : 0.6984
Specificity : 0.9219

Classification Report:
              precision    recall  f1-score   support
           0       0.81      0.92      0.86        64
           1       0.81      0.61      0.70        36
    accuracy                           0.81       100
   macro avg       0.81      0.77      0.78       100

New user -> class: 0  probability of purchase: 0.1084
```

**Interpretation:** 81 % of the test users were classified correctly, but recall for buyers is only 61 % — the model misses about 39 % of actual purchasers. Precision is 81 %, so when it predicts a purchase it is usually right. Lowering the decision threshold would raise recall at the cost of more false alarms.

---

## Formula sheet (write these in the journal)

```
Accuracy    = (TP + TN) / (TP + TN + FP + FN)
Error rate  = (FP + FN) / (TP + TN + FP + FN) = 1 - Accuracy
Precision   = TP / (TP + FP)      -> of all predicted positive, how many were right
Recall      = TP / (TP + FN)      -> of all actual positive, how many were found
Specificity = TN / (TN + FP)      -> of all actual negative, how many were right
F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
```

`confusion_matrix` from sklearn has **rows = actual**, **columns = predicted**, and `cm.ravel()` returns `TN, FP, FN, TP` in that order for binary classification.
