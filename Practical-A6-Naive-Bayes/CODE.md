# Practical A6 — Data Analytics-III (Naïve Bayes)

**Aim:** Implement the Naïve Bayes classification algorithm on `iris.csv` and evaluate it using the confusion matrix, accuracy, error rate, precision and recall.

**Dataset:** Iris — 150 rows, 4 numeric features, 3 classes (`Setosa`, `Versicolor`, `Virginica`).
**Source:** <https://www.kaggle.com/datasets/uciml/iris>

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Full Code

```python
# ============================================================
# Practical A6 : Naive Bayes classification on the Iris dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ---------------- 1. Load the dataset ----------------
df = pd.read_csv("iris.csv")
print("Shape:", df.shape)          # (150, 5)
print(df.head())
print("\nClass counts:\n", df["variety"].value_counts())

# ---------------- 2. Preprocessing ----------------
print("\nMissing values:\n", df.isnull().sum())

# Independent variables (measurements) and dependent variable (species)
X = df.drop("variety", axis=1)
y = df["variety"]

# ---------------- 3. Train-test split (80 % / 20 %) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

# ---------------- 4. Feature scaling ----------------
# GaussianNB does NOT require scaling, but it is shown here because the
# handout's algorithm step 4 says "perform data preprocessing".
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- 5. Train the Naive Bayes model ----------------
model = GaussianNB()
model.fit(X_train_scaled, y_train)

print("\nClass priors (P(class)):")
for cls, prior in zip(model.classes_, model.class_prior_):
    print(f"  {cls}: {prior:.4f}")

print("\nPer-class mean (theta_) of each feature:")
print(pd.DataFrame(model.theta_, index=model.classes_, columns=X.columns))

print("\nPer-class variance (sigma_) of each feature:")
print(pd.DataFrame(model.var_, index=model.classes_, columns=X.columns))

# ---------------- 6. Predictions ----------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print("\nFirst 5 predictions:", y_pred[:5])
print("Actual values       :", y_test.values[:5])

# ---------------- 7. Confusion matrix ----------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print("\nConfusion Matrix:\n", cm_df)

# For a 3-class problem, TP/FP/FN/TN are computed per class (one-vs-rest)
for i, cls in enumerate(model.classes_):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - TP - FN - FP
    print(f"\nClass {cls}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  Precision = {TP/(TP+FP) if TP+FP else 0:.4f}")
    print(f"  Recall    = {TP/(TP+FN) if TP+FN else 0:.4f}")

# ---------------- 8. Overall metrics ----------------
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n----- Evaluation -----")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Error Rate     : {error_rate:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro)   : {recall_macro:.4f}")
print(f"F1 (macro)       : {f1_macro:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- 9. Plot the confusion matrix ----------------
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------- 10. Predict a new flower ----------------
new_flower = pd.DataFrame([[5.0, 3.4, 1.5, 0.2]], columns=X.columns)
new_flower_scaled = scaler.transform(new_flower)
print("\nPredicted species:", model.predict(new_flower_scaled)[0])
print("Class probabilities:",
      {str(c): round(float(p), 4)
       for c, p in zip(model.classes_, model.predict_proba(new_flower_scaled)[0])})
```

## Output (abridged)

```text
Shape: (150, 5)
Class counts: Setosa 50, Versicolor 50, Virginica 50
Missing values: all 0
Train size: (120, 4)   Test size: (30, 4)

Class priors (P(class)):
  Setosa: 0.3333
  Versicolor: 0.3333
  Virginica: 0.3333

Per-class mean (theta_) of each feature  [standardised units, because of the scaler]
            sepal.length  sepal.width  petal.length  petal.width
Setosa         -1.022989     0.820924     -1.301717    -1.250858
Versicolor      0.105483    -0.667934      0.273971     0.151420
Virginica       0.917506    -0.152990      1.027746     1.099438

Per-class variance (sigma_) of each feature:
            sepal.length  sepal.width  petal.length  petal.width
Setosa          0.132297     0.778332      0.008139     0.022494
Versicolor      0.316001     0.466172      0.061740     0.059118
Virginica       0.652251     0.612039      0.104333     0.122051

Confusion Matrix:
             Setosa  Versicolor  Virginica
Setosa           10           0          0
Versicolor        0           9          1
Virginica         0           0         10

Class Setosa:     TP=10, FP=0, FN=0, TN=20  (Precision 1.0000, Recall 1.0000)
Class Versicolor: TP=9,  FP=0, FN=1, TN=20  (Precision 1.0000, Recall 0.9000)
Class Virginica:  TP=10, FP=1, FN=0, TN=19  (Precision 0.9091, Recall 1.0000)

Accuracy          : 0.9667
Error Rate        : 0.0333
Precision (macro) : 0.9697
Recall (macro)    : 0.9667
F1 (macro)        : 0.9666

New flower [5.0, 3.4, 1.5, 0.2] -> Setosa
Class probabilities: {'Setosa': 1.0, 'Versicolor': 0.0, 'Virginica': 0.0}
```

> `theta_` is in **standardised** units because `StandardScaler` was applied. In raw centimetres, Setosa's mean petal length is 1.46 cm and Virginica's is 5.55 cm — the large separation is why the model makes only one error. If you skip scaling (recommended for GaussianNB), `theta_` will be in cm directly.

---

## Key point about multi-class metrics

`precision_score` / `recall_score` need an `average` argument for 3 classes:

| `average=` | Meaning |
|---|---|
| `None` | one score **per class** |
| `"macro"` | simple average of the per-class scores — every class weighted equally |
| `"weighted"` | average weighted by the number of samples in each class |
| `"micro"` | aggregates all TP/FP/FN globally (equals accuracy for single-label multi-class) |

For a balanced dataset such as Iris (50/50/50 and a stratified split) macro and weighted are almost identical; use **macro** in the journal.
