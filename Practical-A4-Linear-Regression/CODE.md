# Practical A4 — Data Analytics-I (Linear Regression)

**Aim:** Build a Linear Regression model in Python to predict house prices (`medv`) using the Boston Housing dataset, and evaluate it with MSE, RMSE and R².

**Dataset:** Boston Housing — 506 rows × 14 columns.
**Source:** <https://www.kaggle.com/c/boston-housing>
**File used:** `Assignment-A4-BostonHousing.csv` (the `medv` column is the target)

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Full Code

```python
# ============================================================
# Practical A4 : Linear Regression on Boston Housing dataset
# Target column: medv (Median value of owner-occupied homes)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- 1. Load the dataset ----------------
df = pd.read_csv("Assignment-A4-BostonHousing.csv")
print("Shape:", df.shape)          # (506, 14)
print("\nFirst 5 rows:\n", df.head())

# ---------------- 2. Explore the data ----------------
print("\nColumns:\n", df.columns)
print("\nInfo:")
df.info()
print("\nDescription:\n", df.describe())

# Missing values
print("\nMissing values per column:\n", df.isnull().sum())

# ---------------- 3. Correlation heatmap ----------------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Boston Housing features")
plt.tight_layout()
plt.show()

# Strongest correlations with the target
print("\nCorrelation with medv:\n",
      df.corr()["medv"].sort_values(ascending=False))

# ---------------- 4. Separate features and target ----------------
X = df.drop("medv", axis=1)     # independent variables (13 features)
y = df["medv"]                  # dependent variable  (price)

# ---------------- 5. Train-test split (80 % / 20 %) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining rows:", X_train.shape[0], " Testing rows:", X_test.shape[0])

# ---------------- 6. Train the Linear Regression model ----------------
model = LinearRegression()
model.fit(X_train, y_train)     # learns coefficients by least squares

print("\nIntercept (b0):", model.intercept_)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nCoefficients (b1..b13):\n", coef_df)

# ---------------- 7. Predict on the test set ----------------
y_pred = model.predict(X_test)

# ---------------- 8. Evaluate the model ----------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# SST = total sum of squares, SSE = residual sum of squares,
# SSR = regression (explained) sum of squares;  SST = SSR + SSE
sst = np.sum((y_test - y_test.mean()) ** 2)
sse = np.sum((y_test - y_pred) ** 2)
ssr = sst - sse          # explained part = SST - SSE (makes R2 = SSR/SST exact)

print("\n----- Evaluation -----")
print(f"SST : {sst:.2f}")
print(f"SSE : {sse:.2f}")
print(f"SSR : {ssr:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.4f}")
print("R2 check (SSR/SST):", round(ssr / sst, 4))

# ---------------- 9. Plots ----------------
# Actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2)
plt.xlabel("Actual Prices (medv)")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

# Residual plot (should look like random noise around 0)
residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual (actual - predicted)")
plt.show()

# ---------------- 10. Predict the price of one new house ----------------
new_house = X.iloc[[0]]          # use the first row as a "new" sample
print("\nActual price   :", y.iloc[0])
print("Predicted price:", model.predict(new_house)[0])

# ---------------- Bonus: metrics for the handout's 5-student table ----------------
x = np.array([95, 85, 80, 70, 60])
y_true = np.array([85, 95, 70, 65, 70])

b1 = np.sum((x - x.mean()) * (y_true - y_true.mean())) / np.sum((x - x.mean()) ** 2)
b0 = y_true.mean() - b1 * x.mean()
y_hat = b0 + b1 * x

sst = np.sum((y_true - y_true.mean()) ** 2)
sse = np.sum((y_true - y_hat) ** 2)
ssr = np.sum((y_hat - y_true.mean()) ** 2)

print(f"\nb0 = {b0:.3f}, b1 = {b1:.3f}")
print(f"SST = {sst:.2f}, SSE = {sse:.2f}, SSR = {ssr:.2f}")
print(f"MSE = {sse / len(y_true):.2f}, RMSE = {np.sqrt(sse / len(y_true)):.2f}")
print(f"R2  = {ssr / sst:.3f}")
```

## Output (abridged)

```text
Shape: (506, 14)
Training rows: 404   Testing rows: 102

Intercept (b0): 30.24675

Correlation with medv (strongest first):
rm         0.695360
zn         0.360445
b          0.333461
dis        0.249929
chas       0.175260
age       -0.376955
rad       -0.381626
crim      -0.388305
nox       -0.427321
tax       -0.468536
indus     -0.483725
ptratio   -0.507787
lstat     -0.737663

Coefficients:
     Feature  Coefficient
4       nox   -17.202633
5        rm     4.438835
3      chas     2.784438
7       dis    -1.447865
10  ptratio    -0.915456
12    lstat    -0.508571
8       rad     0.262430
0      crim    -0.113056

----- Evaluation -----
SST : 7480.05
SSE : 2477.69
SSR : 5002.35
MSE : 24.29
RMSE: 4.93
MAE : 3.19
R2  : 0.6688
R2 check (SSR/SST): 0.6688

Actual price   : 24.0
Predicted price: 29.95

b0 = 26.781, b1 = 0.644
SST = 630.00, SSE = 327.40, SSR = 302.60
MSE = 65.48, RMSE = 8.09
R2  = 0.480
```

> The `corr()["medv"]` output is a correlation series (values between −1 and 1), while the coefficient table shows regression slopes. Do not mix the two in the journal — the code prints both, and only the coefficient table is the model equation.

---

## Expected values (write these in the journal)

| Metric | Approximate value |
|---|---|
| Intercept | 30.25 (depends on scikit-learn version) |
| MSE | ≈ 24.3 |
| RMSE | ≈ 4.93 |
| R² | ≈ 0.67 |

**Interpretation to write:** R² ≈ 0.67 means about 67 % of the variance in house prices is explained by the 13 features — a moderate fit, not perfect. RMSE ≈ 4.93 means predictions are off by about $4,930 on average.
