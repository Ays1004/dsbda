# Practical A2 — Explanation & Viva Questions

## 1. What the practical is asking

The handout has exactly **three** requirements, and the journal should be divided into three parts:

1. **Missing values + inconsistencies** — find them, then fix them with a *justified* technique.
2. **Outliers in numeric variables** — detect them (IQR / boxplot / Z-score) and treat them (cap or remove).
3. **One data transformation** — and *state the reason*: change of scale, linearise a non-linear relation, or reduce skewness toward normality.

The dataset is not given, so you are expected to **create** an "Academic Performance" dataset (marks, attendance, age, grade). Mentioning that it is synthetic is perfectly acceptable — the handout says "Create an Academic performance dataset".

---

## 2. Line-by-line explanation

### Creating the dataset

```python
np.random.seed(42)
```

Fixes the random number generator so the same "random" dataset is produced on every run. Without it, your output will not match your journal.

```python
"Attendance": np.random.randint(55, 101, n)
```

`randint(low, high, size)` → integers from `low` to `high-1`. So this gives 55…100.

```python
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
```

`axis=1` means "sum across the columns of each row". `axis=0` would sum each column.

```python
df["Grade"] = df["Percentage"].apply(assign_grade)
```

`Series.apply()` calls the function once per element. This is the pandas equivalent of a `for` loop and is much shorter.

### Injecting problems on purpose

```python
df.loc[[3, 11], "Age"] = np.nan
```

`loc[row_labels, column_label]` — here rows 3 and 11 get a missing age. `np.nan` (Not a Number) is pandas' standard missing-value marker for numeric columns; `None` is used for object columns.

```python
df.loc[15, "Maths"] = 150        # impossible marks
df.loc[5, "Gender"] = "M"        # inconsistent label
```

A missing value is *absent*, an inconsistency is *present but wrong*. Interviewers like this distinction: 150 marks is not missing, it is **invalid**; "M" is not missing, it is an **inconsistent representation** of "Male".

### Scanning

```python
df.isnull().sum()
df[df.isnull().any(axis=1)]
df[(df["Maths"] < 0) | (df["Maths"] > 100)]
df[~df["Grade"].isin(["A", "B", "C", "D", "F"])]
```

- `df.isnull().any(axis=1)` → `True` for a **row** if any column in that row is null.
- `~` is logical NOT, so `~df["Grade"].isin(valid_list)` filters out every valid grade, leaving only the invalid one.
- `|` is element-wise OR; brackets around each condition are compulsory.

### Fixing missing values

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
```

- **Median** for numeric columns when outliers/skew are present (mean would be pulled by `Age = 45`).
- **Mode** (`[0]` because `mode()` returns a Series) for categorical columns.
- Alternative: `dropna()`, `ffill()`, `bfill()`, `interpolate()`, or a model-based imputer.

### Fixing inconsistencies

```python
df["Maths"] = df["Maths"].clip(0, 100)
df["Gender"] = df["Gender"].replace({"M": "Male"})
```

`clip(lower, upper)` forces every value into the valid range — values above 100 become 100, below 0 become 0. This is called **Winsorising** when applied with percentile fences.

```python
df["Total"] = ...
df["Percentage"] = ...
df["Grade"] = df["Percentage"].apply(assign_grade)
```

**Important:** after changing a source column you must recompute every derived column, otherwise `Percentage = 130` stays in the data. This is exactly the kind of inconsistency the examiner looks for.

### Detecting outliers — IQR method

```python
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Any value outside `[lower, upper]` is an outlier. This is Tukey's rule and it is the method expected in the lab (it matches the boxplot whiskers).

Other methods you should be able to name:

| Method | Rule |
|---|---|
| IQR / boxplot | outside Q1 − 1.5·IQR or Q3 + 1.5·IQR |
| Z-score | \|z\| > 3, where z = (x − μ)/σ |
| Percentile | outside 1st–99th percentile |
| Scatter / Mahalanobis | multivariate distance |

### Treating outliers

Two choices, and you must justify one:

- **Capping / Winsorising** (`clip`) — keeps the row, replaces the extreme value with the fence. Preferred when the sample is small and you cannot afford to lose rows.
- **Trimming / removal** — `df = df[(df[col] >= lower) & (df[col] <= upper)]`. Loses information.

Do **not** replace outliers with the mean — that distorts the distribution and re-introduces the outlier's influence.

### Transformation

```python
df["Attendance_log"] = np.log1p(df["Attendance"])
```

- `log1p(x)` = `log(1 + x)`. The `+1` avoids `log(0) = -∞`, which is why it is used on data that may contain zero.
- A log transform **compresses large values more than small ones**, pulling a long right tail toward the centre → reduces positive skew.
- `sqrt` is a milder version of the same idea.

```python
df["Attendance_MinMax"] = (df["Attendance"] - df["Attendance"].min()) / (
    df["Attendance"].max() - df["Attendance"].min()
)
```

Maps the column to [0, 1]. Reason to write: *"to change the scale for better understanding/comparison of the variable."*

```python
df["Attendance"].skew()
```

Skewness = 0 → symmetric; > 0 → right tail; < 0 → left tail. Comparing skew before and after proves the transformation worked. (Rule of thumb: |skew| < 0.5 is roughly symmetric.)

---

## 3. Journal questions from the handout (with answers)

**Q1. Explain the methods to detect outliers.**
(i) **Boxplot / IQR** — values outside Q1 − 1.5·IQR and Q3 + 1.5·IQR. (ii) **Z-score** — |x − μ|/σ > 3. (iii) **Percentile method** — outside the 1st and 99th percentile. (iv) **Scatter plot** — visual detection for two variables. (v) **Mahalanobis distance / Isolation Forest / DBSCAN** — multivariate and ML-based methods. The lab expects the IQR/boxplot method.

**Q2. Explain data transformation methods.**
(a) **Scaling** — Min-Max to [0,1], Standardisation to mean 0/σ 1. (b) **Non-linear to linear** — log, square root, reciprocal, Box-Cox, log-log. (c) **Distribution shaping** — log/sqrt to remove skew, Box-Cox to approach normality. (d) **Encoding** — label, one-hot. (e) **Discretisation** — binning continuous values into intervals. (f) **Aggregation** — replacing raw values by group means/sums.

**Q3. Algorithm to display statistics of null values.**
```
Step 1: Import pandas, load the dataset into a DataFrame df.
Step 2: Call df.isnull()  -> Boolean DataFrame, True = missing.
Step 3: Call df.isnull().sum() -> count of missing values per column.
Step 4: Call df.isnull().sum().sum() -> total missing values in the dataset.
Step 5: Optional: df.isnull().mean()*100 -> percentage missing per column.
Step 6: Optional: df[df.isnull().any(axis=1)] -> display the affected rows.
Step 7: Print the results.
```

**Q4. Algorithm to replace an outlier with the mean of the variable.**
```
Step 1: Import pandas/numpy and load the data.
Step 2: Compute Q1 = df[col].quantile(0.25), Q3 = df[col].quantile(0.75), IQR = Q3 - Q1.
Step 3: lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR.
Step 4: Compute mean = df[col].mean()  (before replacing).
Step 5: df.loc[(df[col] < lower) | (df[col] > upper), col] = mean.
Step 6: Print/plot the column again to verify.
```
(Write this, but also add the note that the median is statistically safer than the mean.)

---

## 4. Viva questions the external examiner may ask

**1. Difference between missing value and inconsistency?**
Missing = value absent (`NaN`/`None`). Inconsistent = value present but invalid or in a different format (`150` marks, `"M"` instead of `"Male"`, negative age, `"Z"` grade).

**2. Why median and not mean for imputation?**
The mean is pulled by extreme values; the median is a robust statistic and represents the centre of the data even with skew/outliers.

**3. When would you delete rows instead of imputing?**
When the missing fraction is very small (< 1–5 %), when missingness is completely at random (MCAR), or when the column is not important. Deleting > 5–10 % of rows biases the dataset.

**4. What is MCAR, MAR, MNAR?**
Missing Completely At Random (no pattern), Missing At Random (missingness explained by other observed variables), Missing Not At Random (missingness depends on the missing value itself — e.g. high earners hide income). MNAR is the dangerous case.

**5. What is Winsorising?**
Replacing extreme values with a fixed percentile (or the IQR fence) instead of deleting them, so the row is retained but its influence is limited.

**6. Why 1.5 × IQR specifically?**
Tukey chose 1.5 because for a normal distribution it flags roughly 0.7 % of points as outliers — a practical balance between missing real anomalies and flagging normal variation. 3 × IQR marks "extreme" outliers.

**7. What is the difference between an outlier and an error?**
An outlier is a genuine but unusual observation; an error is an impossible value (age 200, marks 150). Errors must be corrected/removed; outliers may be legitimate and should be treated carefully.

**8. Is every outlier useless?**
No. In fraud detection, network intrusion or disease detection the outlier *is* the signal. Removing it blindly destroys the information.

**9. What is skewness? What does a positive value mean?**
A measure of asymmetry of a distribution. Positive skew → long right tail (mean > median), e.g. income. Negative skew → long left tail.

**10. How does a log transform reduce skew?**
It is a concave function, so large values are compressed much more than small ones; the long right tail is pulled in, making the distribution more symmetric/normal.

**11. Which transformations can handle zero or negative values?**
`log1p` handles zeros. For negatives use Yeo-Johnson (Box-Cox requires strictly positive data), or shift the data by a constant before logging.

**12. Min-Max vs Z-score standardisation — which is affected by outliers?**
Min-Max depends on min and max, so a single outlier squashes all other values. Standardisation uses mean and σ, also affected, but less severely. Both should be applied after outlier treatment; robust scaling (median/IQR) is an alternative.

**13. What is the difference between transformation and normalisation?**
Normalisation (scaling) changes only the range/units. Transformation changes the *shape* of the distribution (log, sqrt, Box-Cox) and can linearise relationships.

**14. Why must you recompute derived columns after cleaning?**
Because they were calculated from the dirty values; otherwise the dataset remains internally inconsistent (Total/Percentage/Grade disagreeing with the subject marks).

**15. What is `np.nan`?**
A floating-point representation of "not a number" used as the standard missing marker. Note `np.nan != np.nan` is `True`, so never compare with `==`; use `isna()`/`isnull()`.

**16. Difference between `fillna` and `replace`?**
`fillna` only fills nulls. `replace` substitutes any specified value/pattern (`"M"` → `"Male"`), whether null or not.

**17. What does `clip()` do?**
Limits values to a lower and upper bound. Values below the lower bound become the lower bound, values above the upper become the upper bound.

**18. What is `apply()` and how is it different from `map()`?**
`apply()` applies a function (possibly row/column-wise, `axis=0/1`) and is more general. `map()` works element-wise on a Series using a dict/function/Series for substitution.

---

## 5. Common mistakes

- Replacing outliers with the **mean** (the handout asks for it, but it re-contaminates the data — mention median/capping as the better practice).
- Treating `150` marks as a missing value and imputing it, instead of treating it as invalid and clipping it.
- Forgetting to re-derive `Total`, `Percentage`, `Grade` after fixing marks.
- Using the mean for imputation when the column has outliers.
- Calling `fillna` without assigning/`inplace=True` — the DataFrame is unchanged.
- Computing IQR before and after capping on the same object without a copy, so the second call changes the "before" statistics.
