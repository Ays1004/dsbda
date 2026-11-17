# Practical A4 — Explanation & Viva Questions

## 1. What the practical is asking

Train a **Linear Regression** model that predicts `medv` (house price in $1000s) from the other 13 columns of the Boston Housing dataset, then evaluate it. The examiner will want:

- the equation of the line/model,
- what the coefficients mean,
- and a correct interpretation of **MSE, RMSE, R²**.

---

## 2. Dataset

| Item | Detail |
|---|---|
| Rows | 506 |
| Columns | 14 (13 features + target `medv`) |
| Target | `medv` — median value of owner-occupied homes, in $1000s |
| Example features | `crim` (crime rate), `rm` (average rooms), `lstat` (% lower status), `ptratio` (pupil-teacher ratio), `tax`, `nox`, `dis` |
| Missing values | None |

Source: `https://www.kaggle.com/c/boston-housing`. Note: this dataset was removed from `sklearn.datasets` in scikit-learn 1.2 because of an ethical issue with the `b` column (it encodes race); loading from CSV is the correct approach in the lab.

---

## 3. Theory in the minimum words needed

**Simple linear regression** (1 feature): `y = b0 + b1·x`
**Multiple linear regression** (13 features): `y = b0 + b1x1 + b2x2 + … + b13x13`

- `b0` = intercept: predicted `y` when all features are 0.
- `b1…b13` = coefficients: change in `y` for a **one-unit** change in that feature, holding others constant.

The model is fitted by **Ordinary Least Squares (OLS)** — choosing coefficients that minimise the **Sum of Squared Errors** `Σ(yᵢ − ŷᵢ)²`. The closed-form solution is `β = (XᵀX)⁻¹Xᵀy`; `sklearn` computes it with least squares (SVD-based), not gradient descent.

**Assumptions of linear regression (LINE):**
1. **L**inearity — the relationship between X and y is linear.
2. **I**ndependence of errors — observations/errors are independent.
3. **N**ormality of errors — residuals are normally distributed.
4. **E**qual variance (homoscedasticity) of residuals.
Plus: no severe **multicollinearity** among features.

---

## 4. Line-by-line explanation

### `df.drop("medv", axis=1)`

`X` = everything except the target; `y` = target. `axis=1` drops a column. This is the standard `X`/`y` split — features must never contain the answer, otherwise the model leaks the target.

### `train_test_split(X, y, test_size=0.2, random_state=42)`

- 80 % rows train (404), 20 % test (102).
- `random_state=42` makes the split reproducible — the same rows every run, needed to reproduce your output.
- The test set is never seen during `fit`, so it simulates unseen houses.

### `model = LinearRegression(); model.fit(X_train, y_train)`

`fit` estimates `b0…b13` by minimising SSE. After fitting, `model.coef_` (array of 13 values) and `model.intercept_` hold them.

Reading the coefficients from this run:

| Feature | Coefficient | Meaning |
|---|---|---|
| `rm` | +4.44 | one extra room → price up ≈ $4,440, others constant |
| `chas` | +2.78 | if the house borders the Charles River → +$2,780 |
| `dis` | −1.45 | farther from employment centres → lower price |
| `lstat` | −0.51 | 1 % more lower-status population → −$510 |

### `y_pred = model.predict(X_test)`

Applies the learned equation to the 102 unseen rows. Output is a continuous number (regression), unlike classification which outputs a class label.

### Metrics

| Metric | Formula | Meaning | Good value |
|---|---|---|---|
| **SST** | `Σ(y − ȳ)²` | total variation in the target | — |
| **SSE** | `Σ(y − ŷ)²` | unexplained variation (residuals) | low |
| **SSR** | `SST − SSE` | variation explained by the model | high |
| **MSE** | `SSE / n` | average squared error, penalises big errors | low |
| **RMSE** | `√MSE` | error in the same unit as `y` ($1000s) | low |
| **MAE** | `mean(|y − ŷ|)` | average absolute error, robust to outliers | low |
| **R²** | `1 − SSE/SST = SSR/SST` | proportion of variance explained | close to 1 |

Observed values:

```
SST = 7480.05,  SSE = 2477.69,  SSR = 5002.36
MSE = 24.29,  RMSE = 4.93,  MAE = 3.19,  R² = 0.6688
```

**Interpretation to say out loud:** "R² = 0.67 → 67 % of the variation in house prices is explained by the model; RMSE = 4.93 → typical prediction error is about $4,930. The model is a moderate fit — some price variation comes from factors not present in the data."

### R² versus Adjusted R²

R² always increases when you add any feature, even a useless one. **Adjusted R²** penalises extra predictors:

`Adj R² = 1 − [(1 − R²)(n − 1)] / (n − k − 1)`

Always report Adjusted R² when comparing models with different numbers of features.

### Residual plot

Plot `residual = y_test − y_pred` against predictions. Desired pattern: random scatter around 0. A funnel shape means heteroscedasticity; a curve means the relationship is non-linear and linear regression is the wrong model.

---

## 5. Worked numerical from the handout (very likely to be asked)

Data:

| Student | X (X std) | Y (XII std) |
|---|---|---|
| 1 | 95 | 85 |
| 2 | 85 | 95 |
| 3 | 80 | 70 |
| 4 | 70 | 65 |
| 5 | 60 | 70 |

`x̄ = 78`, `ȳ = 77`, `n = 5`.

**Step 1 — SST**
```
SST = (85−77)² + (95−77)² + (70−77)² + (65−77)² + (70−77)²
    = 64 + 324 + 49 + 144 + 49 = 630
```

**Step 2 — slope**
```
Σ(x−x̄)(y−ȳ) = 17·8 + 7·18 + 2·(−7) + (−8)(−12) + (−18)(−7) = 470
Σ(x−x̄)²     = 289 + 49 + 4 + 64 + 324 = 730
b1 = 470 / 730 = 0.644
```

**Step 3 — intercept**
```
b0 = ȳ − b1·x̄ = 77 − 0.644(78) = 26.78
Model: ŷ = 26.78 + 0.644x
```

**Step 4 — SSR and SSE**

| x | y | ŷ | residual (y−ŷ) | residual² |
|---|---|---|---|---|
| 95 | 85 | 87.94 | −2.94 | 8.64 |
| 85 | 95 | 81.50 | 13.50 | 182.25 |
| 80 | 70 | 78.28 | −8.28 | 68.56 |
| 70 | 65 | 71.85 | −6.85 | 46.92 |
| 60 | 70 | 65.41 | 4.59 | 21.07 |

```
SSE = 327.4
SSR = SST − SSE = 630 − 327.4 = 302.6
```

**Step 5 — metrics**
```
MSE  = SSE/n   = 327.4/5   = 65.48
RMSE = √65.48  = 8.09
R²   = SSR/SST = 302.6/630 = 0.480  (48 %)
```

**Step 6 — comment on fit:** R² = 0.48 means only 48 % of the variation in XII-standard marks is explained by X-standard marks. The model is a **weak/moderate fit** — more than half the variation is unexplained, so other factors (study time, attendance, difficulty) matter. The slope 0.644 says each extra mark in X standard corresponds to ≈ 0.64 extra marks in XII standard.

---

## 6. Viva questions the external examiner may ask

**1. What is linear regression?**
A supervised learning technique that models the relationship between one or more independent variables and a continuous dependent variable by fitting the best straight line (or hyperplane) that minimises the sum of squared errors.

**2. Difference between simple and multiple linear regression?**
Simple → one independent variable (`y = b0 + b1x`). Multiple → two or more (`y = b0 + b1x1 + … + bnxn`).

**3. What is the cost function / loss function here?**
Mean Squared Error: `J = (1/n)Σ(yᵢ − ŷᵢ)²`. OLS minimises SSE (equivalently MSE, since `n` is constant).

**4. How are the coefficients found?**
Closed-form OLS: `β = (XᵀX)⁻¹Xᵀy`. For very large data, gradient descent is used instead. scikit-learn's `LinearRegression` uses least squares (SVD/pinv), not iterative gradient descent.

**5. What does the intercept mean?**
The predicted value of `y` when every feature is 0. Often it has no physical meaning (a house cannot have 0 rooms), but it is required to position the line.

**6. What does a coefficient of −0.51 for `lstat` mean?**
Holding other features constant, a one-unit increase in `lstat` decreases the predicted `medv` by 0.51 ($510).

**7. What is R²? Range?**
Coefficient of determination — the proportion of total variance in `y` explained by the model. Range 0 to 1 for a reasonable model on training data (can be negative on test data if the model is worse than predicting the mean). 1 = perfect, 0 = no better than the mean.

**8. Difference between R² and Adjusted R²?**
R² increases whenever a feature is added. Adjusted R² penalises each added feature, so it only increases if the new feature improves the model more than chance would.

**9. Difference between MSE, RMSE and MAE?**
MSE averages squared errors (penalises large errors, units squared). RMSE is its square root (same units as target). MAE averages absolute errors and is less sensitive to extreme outliers.

**10. Why is RMSE preferred over MSE when reporting?**
It is in the same unit as the target ($1000s here), so the error is directly interpretable.

**11. What is SST = SSR + SSE?**
Partition of total variation: total (SST) = explained by regression (SSR) + unexplained residuals (SSE). `R² = SSR/SST = 1 − SSE/SST`.

**12. What is the difference between correlation and regression?**
Correlation measures the strength and direction of a linear association (symmetric, unit-free, −1 to 1). Regression models the dependence of `y` on `x` and gives a predictive equation (asymmetric, has units).

**13. What are the assumptions of linear regression?**
Linearity, independence of observations, normality of residuals, homoscedasticity (constant variance), no severe multicollinearity, no influential outliers.

**14. How do you check the assumptions?**
Residual plots (linearity / homoscedasticity), Q-Q plot and Shapiro-Wilk test (normality), Durbin-Watson (independence), VIF > 10 (multicollinearity), Cook's distance (influential points).

**15. What is multicollinearity? How do you detect/remove it?**
Features that are highly correlated with each other, making coefficients unstable. Detect with a correlation heatmap or VIF. Remove one of the correlated features, combine them (PCA), or use ridge regression.

**16. Does linear regression need feature scaling?**
Not for prediction accuracy (OLS is scale-invariant), but scaling helps interpretation, helps regularised versions (ridge/lasso), and speeds up gradient-descent implementations.

**17. What happens if you don't split the data?**
You evaluate on the same data used for training → optimistic, overfitted performance. The test set must be unseen.

**18. What is overfitting? How is it detected here?**
The model memorises training noise and performs poorly on new data. Detect by comparing train vs test R²/RMSE — a large gap indicates overfitting.

**19. Why `random_state=42`?**
To make the random split reproducible: the same rows go to train/test each run, so results and screenshots match.

**20. Which features are most important in this dataset?**
By absolute correlation/coefficient: `rm` (positive, strongest), `lstat` (negative), `ptratio`, `dis`. `crim` and `tax` also matter but are correlated with each other.

**21. Can linear regression be used for classification?**
Not properly — it outputs unbounded continuous values and does not model probabilities. Use logistic regression for classification.

**22. What is regularisation? Ridge vs Lasso?**
Adding a penalty to the loss to reduce overfitting. **Ridge (L2)** shrinks coefficients toward zero; **Lasso (L1)** can set some coefficients exactly to zero (feature selection). **ElasticNet** mixes both.

---

## 7. Common mistakes

- Forgetting to drop the target from `X` (data leakage).
- Reporting only MSE without RMSE/R².
- Saying "R² = 0.67, so the model is 67 % accurate" — R² is **variance explained**, not accuracy.
- Using `model.score(X_test, y_test)` and claiming it is MSE — it returns R².
- Computing SST/SSE on the whole dataset but predictions on the test set only (mismatched arrays).
- Not mentioning that the dataset was removed from sklearn and loaded from CSV.
