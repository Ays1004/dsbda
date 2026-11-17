# Practical A5 — Explanation & Viva Questions

## 1. What the practical is asking

The handout wants exactly two things:

1. **Logistic Regression** on `Social_Network_Ads.csv` to classify `Purchased` (0/1).
2. **Confusion matrix** → TP, FP, TN, FN → accuracy, error rate, precision, recall.

The dataset has `User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`. The target is binary, so this is **classification**, not regression.

---

## 2. Theory: logistic regression

Linear regression gives a value from −∞ to +∞, which cannot be a probability. Logistic regression applies the **sigmoid** to the linear combination:

```
z    = b0 + b1x1 + b2x2 + ... + bnxn
p    = σ(z) = 1 / (1 + e^(−z))
```

Properties of σ(z): output in (0, 1); σ(0) = 0.5; monotonic; derivative `σ'(z) = σ(z)(1 − σ(z))` (needed for gradient descent).

**Decision rule:** predict class 1 if `p ≥ 0.5`, else class 0 (0.5 is the default threshold and can be tuned).

**Log-odds (logit) form — remember this line:** `log(p / (1 − p)) = b0 + b1x1 + ...`. Logistic regression is a *linear model in the log-odds*, which is why the decision boundary is a straight line (a hyperplane).

**Training:** coefficients are found by **maximum likelihood**, which is equivalent to minimising **log loss / binary cross-entropy**:

```
J = −(1/n) Σ [ y·log(p) + (1−y)·log(1−p) ]
```

There is no closed-form solution — `sklearn` uses an iterative solver (lbfgs by default), hence `max_iter=1000`.

**Coefficient interpretation:** a positive coefficient increases the log-odds of `Purchased = 1`, i.e. increases the probability. Odds ratio = `e^coefficient`.

---

## 3. Line-by-line explanation

### Preprocessing

```python
df = df.drop("User ID", axis=1)
```

`User ID` is an arbitrary identifier — it has no predictive meaning and would add noise. Dropping ID/timestamp columns is a standard step.

```python
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
```

Logistic regression is a mathematical model — it needs numbers. Binary categorical → 0/1 mapping (a special case of label/one-hot encoding where one column suffices).

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
```

- 75 % train / 25 % test.
- `stratify=y` keeps the same class ratio (≈64 % / 36 %) in both sets; important for classification with imbalance.
- `random_state` for reproducibility.

### Scaling — the most commonly asked point

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

`Age` ranges roughly 18–60, `EstimatedSalary` 15 000–150 000. Unscaled, salary dominates the gradient and the model converges slowly/poorly. **`fit_transform` on train only, `transform` on test** — fitting the scaler on the test set leaks test statistics into training.

Does logistic regression *need* scaling? Strictly no (the optimum is unchanged), but it improves numerical stability and convergence, and is required for regularised/penalised versions.

### `predict` vs `predict_proba`

- `predict()` → hard class label (0/1) after applying the 0.5 threshold.
- `predict_proba()` → `[P(class 0), P(class 1)]`; column `[:,1]` is the probability of purchase. Needed for ROC-AUC and for changing the threshold.

### Confusion matrix layout

```
                 Predicted 0     Predicted 1
Actual 0    TN = 59           FP = 5
Actual 1    FN = 14           TP = 22
```

- **TP** — predicted purchased, actually purchased.
- **TN** — predicted not purchased, actually not purchased.
- **FP** — predicted purchased, actually not (false alarm, Type I error).
- **FN** — predicted not purchased, actually purchased (miss, Type II error).

`cm.ravel()` returns them in the order `TN, FP, FN, TP` for a binary matrix — memorise this, it is asked constantly.

### The numbers from this run

```
Accuracy = (59+22)/100 = 0.8100   -> 81 % correct
Error    = 1 - 0.81    = 0.1900
Precision= 22/(22+5)   = 0.8148   -> when it says "will buy", it is right 81 % of the time
Recall   = 22/(22+14)  = 0.6111   -> it finds only 61 % of the actual buyers
F1       = 0.6984
Specificity = 59/(59+5) = 0.9219
```

**Observation to write:** accuracy is high mainly because the majority class (not purchased) is easy to predict. Recall for the buyers is only 61 %, i.e. **39 % of potential buyers are missed** — that is the business cost. If the campaign is cheap, lower the threshold (e.g. 0.4) to raise recall at the cost of precision.

---

## 4. Metric definitions + worked example (handout Q3)

Given `TP = 35, FP = 10, FN = 5, TN = 50` (total 100):

```
Accuracy    = (35+50)/100          = 0.85  (85 %)
Error rate  = (10+5)/100 = 1−0.85  = 0.15  (15 %)
Precision   = 35/(35+10)           = 0.778
Recall      = 35/(35+5)            = 0.875
Specificity = 50/(50+10)           = 0.833
F1          = 2·(0.778·0.875)/(0.778+0.875) = 0.824
```

Comment: the model is good at detecting positives (recall 0.875) with acceptable false alarms (precision 0.778); accuracy 85 % is respectable but should always be read together with precision/recall, especially if classes are imbalanced.

---

## 5. Journal questions from the handout (with answers)

**Q1. Explain logistic regression in detail.**
A supervised classification algorithm that models the probability that a sample belongs to the positive class. It computes a linear combination `z = b0 + Σbᵢxᵢ` and passes it through the sigmoid `σ(z) = 1/(1+e^−z)`, giving a value in (0,1). A threshold (default 0.5) converts the probability into a class. Coefficients are learned by maximum likelihood, i.e. minimising log loss. It produces a linear decision boundary and works for binary and (with softmax) multi-class problems.

**Q2. Differentiate linear vs logistic regression.**

| Point | Linear Regression | Logistic Regression |
|---|---|---|
| Output | continuous value | probability / class |
| Function | `y = b0 + Σbᵢxᵢ` | `p = σ(b0 + Σbᵢxᵢ)` |
| Range | −∞ to +∞ | 0 to 1 |
| Loss | MSE / SSE | log loss (cross-entropy) |
| Use | regression (price, temperature) | classification (spam, purchase) |
| Assumptions | linearity, normal residuals | linearity in log-odds, independent observations |
| Evaluation | R², RMSE | accuracy, precision, recall, F1, ROC-AUC |

**Q3. TP/FP/TN/FN + metrics** → see the worked example in section 4.

**Q4. Comment on whether the model is best fit.**
With accuracy 0.81, precision 0.81, recall 0.61 and F1 0.70, the model is a *reasonable but not optimal* fit. It is conservative: it predicts "not purchased" too often, so recall is low. Improvements: tune the decision threshold, add features, use class weights (`class_weight="balanced"`), or try a non-linear model.

**Q5. Write the preprocessing code of step 4 and explain.**
```python
df = pd.read_csv("Assignment-A5-Social_Network_Ads.csv")
df = df.drop("User ID", axis=1)                         # remove identifier
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})  # encode category
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
Explanation: load → drop irrelevant ID → encode gender to numeric → separate features/target → stratified split → standardise using training statistics only. Each step is needed to make the data numeric, leak-free and on a comparable scale.

---

## 6. Viva questions the external examiner may ask

**1. Why can't we use linear regression for classification?**
It outputs unbounded values, does not give probabilities, is sensitive to outliers, and a straight line cannot model a 0/1 step. It can also produce predictions outside [0,1].

**2. What is the sigmoid function and why is it used?**
`σ(z) = 1/(1+e^−z)`; it squashes any real number into (0,1), which can be interpreted as a probability. It is differentiable, so gradient-based optimisation works.

**3. What is the decision boundary of logistic regression?**
The set of points where `p = 0.5`, i.e. `b0 + Σbᵢxᵢ = 0` — a straight line (hyperplane) in feature space. Hence logistic regression is a **linear classifier**.

**4. What loss function does it use and why not MSE?**
Log loss / binary cross-entropy. With the sigmoid, MSE produces a non-convex cost surface with many local minima and vanishing gradients; log loss is convex in the parameters and gives a larger gradient when the prediction is confidently wrong.

**5. What is a confusion matrix?**
A table comparing actual vs predicted classes; for binary it has TP, TN, FP, FN. It shows *types* of errors, which accuracy alone hides.

**6. Is high accuracy always good?**
No. With 95 % negatives, predicting "negative" always gives 95 % accuracy but zero recall. Always check precision, recall, F1 and the class balance.

**7. Precision vs recall — when to prefer which?**
Precision when false positives are costly (spam filter, innocent user flagged as fraud). Recall when false negatives are costly (cancer detection, fraud missed). F1 balances both.

**8. What is the F1 score?**
Harmonic mean of precision and recall: `2PR/(P+R)`. It penalises extreme imbalance between the two.

**9. What is specificity?**
`TN/(TN+FP)` — the true negative rate, ability to correctly reject negatives.

**10. How does `stratify=y` help?**
It preserves the class proportions in train and test, so the test set represents the real distribution and metrics are reliable.

**11. Why scale features? Which scaler?**
Because salary ≫ age in magnitude; gradient descent converges faster and regularisation is fair across features. `StandardScaler` (mean 0, σ 1) is typical; `MinMaxScaler` also works.

**12. What is `max_iter`?**
The maximum number of solver iterations. If it is too small, sklearn warns "lbfgs failed to converge" and coefficients are not optimal.

**13. What does `predict_proba` return?**
An `n × k` array of class probabilities that sum to 1 per row. For binary, column 1 is `P(y = 1)`.

**14. Can you change the 0.5 threshold? Why?**
Yes. `y_pred = (y_prob >= 0.4).astype(int)`. Lower threshold → higher recall, lower precision; higher threshold → higher precision, lower recall. Choose based on business cost.

**15. What is ROC-AUC?**
ROC plots True Positive Rate vs False Positive Rate at all thresholds. AUC is the probability that a random positive is ranked above a random negative. 0.5 = random, 1.0 = perfect; it is threshold-independent.

**16. What is class imbalance and how do you handle it?**
One class far outnumbers the other. Handle with resampling (SMOTE, undersampling), `class_weight="balanced"`, threshold tuning, or metrics other than accuracy.

**17. What is regularisation in logistic regression?**
A penalty on large coefficients (L2 by default in sklearn, controlled by `C` = 1/λ). It prevents overfitting. Smaller `C` = stronger regularisation.

**18. What is multinomial vs one-vs-rest logistic regression?**
One-vs-rest trains one binary classifier per class. Multinomial (softmax) trains a single model producing a probability distribution over all classes; preferred when classes are mutually exclusive.

**19. What are odds and log-odds?**
Odds = `p/(1−p)`. Log-odds (logit) = `log(p/(1−p))`, which is exactly the linear predictor. A coefficient of 0.7 means the odds multiply by `e^0.7 ≈ 2` for a one-unit increase.

**20. Why is logistic regression called a "regression" if it classifies?**
Because it estimates a continuous quantity — the log-odds/probability — and only the final threshold turns it into a class.

**21. What happens with perfectly separable data?**
The coefficients diverge to infinity (no finite maximum likelihood); regularisation or more data is needed. sklearn may warn about convergence.

---

## 7. Common mistakes

- Fitting the scaler on the full dataset (`fit_transform(X)`) instead of only on train → data leakage.
- Forgetting `stratify=y` and getting an unbalanced test set.
- Reading the confusion matrix in the wrong order (rows are **actual**, columns are **predicted**).
- Mixing up precision and recall in the viva.
- Reporting accuracy alone on an imbalanced dataset.
- Leaving `User ID` in `X` — it becomes a meaningless but high-impact feature.
- Using `LinearRegression` for a 0/1 target.
