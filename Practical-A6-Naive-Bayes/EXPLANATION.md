# Practical A6 — Explanation & Viva Questions

## 1. What the practical is asking

Implement **Naïve Bayes** on `iris.csv` and evaluate it. Two things are being tested:

1. Do you understand **Bayes' theorem** and the "naïve" independence assumption?
2. Can you compute and interpret a **confusion matrix** for a 3-class problem (TP/FP/FN/TN are per class here)?

The handout also expects you to solve the **car theft** example by hand — that is almost guaranteed to be asked.

---

## 2. Theory

### Bayes' theorem

```
P(A|B) = P(B|A) · P(A) / P(B)
```

In classification terms:

```
P(class | features) = P(features | class) · P(class) / P(features)
```

- `P(class)` = **prior** — how common the class is before seeing the features.
- `P(features | class)` = **likelihood**.
- `P(class | features)` = **posterior** — what we want.
- `P(features)` = **evidence** — constant for all classes, so it can be ignored when comparing.

**Prediction:** choose the class with the highest posterior (MAP rule). Because `P(features)` is the same for every class, we only compare `likelihood × prior`.

### The "naïve" assumption

Computing `P(x1, x2, …, xn | class)` jointly is impossible with limited data. Naïve Bayes assumes the features are **conditionally independent given the class**:

```
P(x1, x2, …, xn | class) = P(x1|class) · P(x2|class) · … · P(xn|class)
```

This assumption is usually false in reality (petal length and petal width are correlated), yet the classifier still works well — that is the "naïve but effective" point to say in the viva.

### Gaussian Naïve Bayes (used for Iris)

Iris features are continuous, so `P(xᵢ | class)` is modelled as a **normal distribution** per class and feature:

```
P(xᵢ | c) = (1 / √(2πσ²_c,i)) · exp( −(xᵢ − μ_c,i)² / (2σ²_c,i) )
```

`sklearn` estimates `μ` (`theta_`) and `σ²` (`var_`) from the training data, then multiplies the per-feature densities with the prior. In practice it works in **log space** to avoid floating-point underflow:

```
log posterior = log prior + Σ log P(xᵢ | class)
```

### Laplace smoothing

If a feature value never appears with a class, the likelihood becomes 0 and the whole product collapses to 0. Add 1 to every count:

```
P(xᵢ|c) = (count(xᵢ, c) + 1) / (count(c) + k)     (k = number of distinct values)
```

This matters for **Multinomial/Bernoulli** NB (text). `GaussianNB` does not need it, but the examiner still asks.

**Variants to name:** Gaussian (continuous features), Multinomial (word counts), Bernoulli (binary features), Complement (imbalanced text).

---

## 3. Line-by-line explanation

### `GaussianNB()`

Chosen automatically because all four Iris features are continuous. For text counts you would use `MultinomialNB()`.

### Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**GaussianNB does not require scaling** — it estimates mean and variance per feature, and the likelihood is invariant to a linear rescaling of that feature. It is included only because the handout's algorithm step says "perform data preprocessing". Be ready to say: *"Scaling is not necessary for Naïve Bayes; it is done here for consistency with the algorithm steps."*

### `model.class_prior_`

`[1/3, 1/3, 1/3]` for Iris because the training set is stratified (50 samples per class). On an imbalanced dataset these would be the observed frequencies.

### `model.theta_` and `model.var_`

`theta_` is a 3×4 matrix — mean of each feature for each class. `var_` is the variance matrix. Example: Setosa has mean petal length ≈ 1.46 cm with variance ≈ 0.03, while Virginica's mean is ≈ 5.55 cm — the classes are well separated, which is why the accuracy is high.

### Predictions

`predict()` returns the class with the highest posterior. `predict_proba()` returns the three posteriors; the code predicts Setosa with probability ≈ 1.0 for `[5.0, 3.4, 1.5, 0.2]` because that measurement is deep inside the Setosa cluster.

### Confusion matrix for 3 classes

```
                 Pred Setosa  Pred Versicolor  Pred Virginica
Actual Setosa         10            0                0
Actual Versicolor      0            9                1
Actual Virginica       0            1               10
```

For multi-class, TP/FP/FN/TN are computed **one class vs the rest**:

```
TP(class i) = cm[i, i]
FN(class i) = row i sum − TP
FP(class i) = column i sum − TP
TN(class i) = total − TP − FN − FP
```

For Versicolor: TP = 9, FN = 1, FP = 0, TN = 20.

### Metrics from this run

```
Accuracy = (10+9+10)/30 = 0.9667  (29 of 30 correct)
Error    = 0.0333
Precision (macro) = 0.9697
Recall (macro)    = 0.9667
F1 (macro)        = 0.9666
```

The single error is a Versicolor classified as Virginica — the two species overlap in petal measurements. **Setosa is linearly separable** and is always classified perfectly.

---

## 4. Worked car theft example (handout Q2)

| # | Colour | Type | Origin | Stolen |
|---|---|---|---|---|
| 1 | Red | Sports | Domestic | Yes |
| 2 | Red | Sports | Domestic | No |
| 3 | Red | Sports | Domestic | Yes |
| 4 | Yellow | Sports | Domestic | No |
| 5 | Yellow | Sports | Imported | Yes |
| 6 | Yellow | SUV | Imported | No |
| 7 | Yellow | SUV | Imported | Yes |
| 8 | Yellow | SUV | Domestic | No |
| 9 | Red | SUV | Imported | No |
| 10 | Red | Sports | Imported | Yes |

Total = 10, `Yes` = 5, `No` = 5 → `P(Yes) = P(No) = 0.5`.

Query: **Colour = Red, Type = SUV, Origin = Domestic**

**For Yes:**
```
P(Red    | Yes) = 3/5 = 0.6     (rows 1, 3, 10)
P(SUV    | Yes) = 1/5 = 0.2     (row 7)
P(Domestic| Yes) = 2/5 = 0.4    (rows 1, 3)
P(X | Yes)·P(Yes) = 0.6 × 0.2 × 0.4 × 0.5 = 0.024
```

**For No:**
```
P(Red    | No) = 2/5 = 0.4      (rows 2, 9)
P(SUV    | No) = 2/5 = 0.4      (rows 6, 8)
P(Domestic| No) = 4/5 = 0.8     (rows 2, 4, 6, 8)
P(X | No)·P(No) = 0.4 × 0.4 × 0.8 × 0.5 = 0.064
```

**Conclusion:** `0.064 (No) > 0.024 (Yes)` → the car is **not stolen**.

Optional normalisation (to turn scores into probabilities):
```
P(No | X)  = 0.064 / (0.064 + 0.024) = 0.727
P(Yes | X) = 0.024 / 0.088           = 0.273
```

This example also shows the value of **Laplace smoothing**: if some combination had zero counts, the product would become zero without it.

---

## 5. Journal questions from the handout (with answers)

**Q1. Explain the concepts of the Naïve Bayes classifier.**
It is a probabilistic supervised classifier based on Bayes' theorem. It calculates the posterior probability of each class given the features, assuming conditional independence between features. The class with the highest posterior wins. It is fast, works well with high-dimensional data (text), needs little training data, and handles both binary and multi-class problems. Its weakness is the independence assumption and the zero-frequency problem (solved by Laplace smoothing).

**Q2. Car theft probability** → see the worked answer above.

**Q3. Write the preprocessing code of step 4 and explain.**
```python
df = pd.read_csv("iris.csv")
print(df.isnull().sum())          # no missing values
X = df.drop("variety", axis=1)    # 4 numeric features
y = df["variety"]                 # 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Explanation: load the CSV; verify there are no missing values; separate the four measurements from the species label; do a stratified 80/20 split to preserve the 50/50/50 class ratio; standardise using training statistics only. GaussianNB does not need scaling, but it satisfies the preprocessing step and does no harm.

---

## 6. Viva questions the external examiner may ask

**1. State Bayes' theorem.**
`P(A|B) = P(B|A)·P(A)/P(B)`. In classification: posterior = likelihood × prior / evidence.

**2. Why is it called "naïve"?**
Because it assumes all features are conditionally independent given the class — an assumption that is almost never true (e.g. petal length and petal width are correlated). The assumption simplifies computation dramatically.

**3. What are prior, likelihood, posterior and evidence?**
Prior `P(class)` — belief before seeing data; likelihood `P(x|class)` — how probable the features are for that class; posterior `P(class|x)` — updated belief; evidence `P(x)` — normalising constant, identical across classes.

**4. Why can the denominator P(x) be ignored?**
It does not depend on the class, so it is the same for every class. Comparing numerators is enough to pick the maximum.

**5. Why is GaussianNB used for Iris?**
Because all features are continuous numeric values, so each feature's likelihood is modelled with a Gaussian distribution per class. Multinomial/Bernoulli NB are for counts/binary features.

**6. What are `theta_` and `var_` in sklearn's GaussianNB?**
`theta_` = per-class feature means; `var_` = per-class feature variances. They parameterise the Gaussian likelihoods.

**7. What is Laplace smoothing and why is it needed?**
Adding a small constant (usually 1) to all counts so that a feature–class combination never seen in training does not get probability 0, which would zero out the entire product. Formula: `(count + 1)/(total + k)`.

**8. What is the zero-frequency problem?**
If a categorical value never occurs with a class in training, its likelihood is 0, forcing the posterior to 0 regardless of all other evidence. Laplace smoothing fixes it.

**9. Why work in log space?**
Multiplying many small probabilities underflows to 0. Summing logs keeps values numerically stable and converts multiplication to addition.

**10. Is Naïve Bayes a linear or non-linear classifier?**
For GaussianNB with shared covariance it is a linear classifier; with different variances per class the decision boundary is quadratic. MultinomialNB is linear in log-counts.

**11. Does Naïve Bayes need feature scaling?**
No. It estimates per-feature means/variances (or counts), so monotonic rescaling does not change the decision. It is scale-invariant.

**12. Can Naïve Bayes handle missing values?**
Not directly — sklearn's implementations do not accept NaN. Impute first, or treat "missing" as its own category for categorical NB.

**13. Advantages of Naïve Bayes?**
Very fast, low memory, works with small datasets, handles many features/classes well, naturally multi-class, good baseline for text classification.

**14. Disadvantages?**
The independence assumption is unrealistic; probability estimates are poorly calibrated (overconfident, pushed toward 0/1); sensitive to correlated features; the zero-frequency problem.

**15. Difference between Naïve Bayes and logistic regression?**
NB is **generative** (models P(x|y) and P(y)); logistic regression is **discriminative** (models P(y|x) directly). NB converges faster with little data but its independence assumption is stronger; logistic regression is usually more accurate with enough data.

**16. How do you compute TP/FP/FN/TN for a 3-class problem?**
One class vs the rest. TP = diagonal cell of that class; FN = sum of its row minus TP; FP = sum of its column minus TP; TN = total minus TP − FN − FP.

**17. What does `average="macro"` do?**
Computes precision/recall/F1 separately for each class and takes the unweighted mean — every class counts equally, which highlights poor performance on a minority class.

**18. Which class was misclassified in this practical and why?**
One Versicolor was predicted as Virginica. The two species overlap in petal length/width, so their Gaussian distributions overlap; Setosa is well separated and was classified with 100 % accuracy.

**19. What is the difference between `predict` and `predict_proba`?**
`predict` returns the MAP class label after comparing posteriors; `predict_proba` returns the posterior probability of each class.

**20. Why is Naïve Bayes popular for spam filtering?**
Text has thousands of features; the independence assumption is a decent approximation for word occurrences, training is extremely fast, and Multinomial/Bernoulli NB perform very well with small labelled datasets.

**21. What would you do to improve the model's accuracy on Iris?**
Use all 150 samples with cross-validation, remove scaling (irrelevant), tune `var_smoothing`, compare with KNN/SVM/decision tree, and use stratified k-fold to get a stable estimate.

---

## 7. Common mistakes

- Reversing the confusion matrix orientation (rows = actual, columns = predicted).
- Using `accuracy` alone for multi-class and forgetting per-class precision/recall.
- Calling `precision_score` without `average=` for 3 classes → sklearn raises a warning/error.
- Thinking scaling is mandatory for Naïve Bayes.
- Forgetting to add Laplacian smoothing in the hand-written example.
- Mixing up `P(A|B)` with `P(B|A)` in the car theft example (that inversion is exactly what the examiner checks).
