# Practical A10 — Explanation & Viva Questions

## 1. What the practical is asking

Four deliverables:

1. **List the features and their types** (numeric vs nominal).
2. A **histogram for each feature**.
3. A **box plot for each feature**.
4. **Compare distributions and identify outliers.**

This is pure univariate EDA. The marks are in the interpretation, not the plotting calls.

---

## 2. Features and their types (deliverable 1)

| Feature | Type | Measurement |
|---|---|---|
| `sepal.length` | numeric, continuous (ratio scale) | sepal length in cm |
| `sepal.width` | numeric, continuous | sepal width in cm |
| `petal.length` | numeric, continuous | petal length in cm |
| `petal.width` | numeric, continuous | petal width in cm |
| `variety` | nominal categorical (3 levels) | species: Setosa, Versicolor, Virginica |

There are **150 samples**, 50 per species. No missing values.

If the examiner asks why `variety` is nominal and not ordinal: the three species have no natural order — you cannot say Setosa < Versicolor < Virginica in any meaningful sense.

---

## 3. Line-by-line explanation

### `df.columns[:-1]`

All columns except the last (`variety`), i.e. the four numeric features. Slicing a list/Index with `[:-1]` drops the last element. This assumes `variety` is the last column — true for this CSV; a safer version is `df.select_dtypes(include="number").columns`.

### Histograms

```python
ax.hist(df[col], bins=15, edgecolor="black", color="skyblue")
```

- `bins=15` → the range is divided into 15 equal-width intervals; bar height = number of observations in that interval.
- `edgecolor` makes the bars distinguishable.
- A histogram shows **shape**: symmetric, skewed, bimodal, uniform.

```python
sns.histplot(data=df, x=col, hue="variety", bins=15, element="step")
```

Adding `hue="variety"` splits each bin by species. This is the plot that shows **why the classes are separable**: petal length/width are clearly separated for Setosa and mostly separated for the other two.

**Choosing bins:** too few bins hide the shape (e.g. 5 bins merge the two petal clusters); too many produce a spiky, noisy plot. 10–20 is a reasonable practical choice for n = 150.

### Box plots

```python
sns.boxplot(y=df[col], color="salmon")
```

One box per feature. Shows median, Q1–Q3, whiskers at 1.5 × IQR and individual outliers.

```python
sns.boxplot(data=df, x="variety", y=col, hue="variety", palette="Set2", legend=False)
```

Grouped version: one box per species. `hue="variety"` is required in newer seaborn to use a palette without a warning; `legend=False` avoids a redundant legend because the x-axis already shows the species.

### `df[features].agg(["mean", "median", "std", "skew"])`

Produces one table with four statistics per feature. This is the numeric backup for the visual comparison — always quote a number along with the picture.

### IQR outlier detection

```python
Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df[col] < lower) | (df[col] > upper)][col]
```

Exact numbers from the run:

| Feature | Q1 | Q3 | IQR | Accepted range | Outliers |
|---|---|---|---|---|---|
| sepal.length | 5.10 | 6.40 | 1.30 | [3.15, 8.35] | none |
| sepal.width | 2.80 | 3.30 | 0.50 | [2.05, 4.05] | 4.4, 4.1, 4.2, 2.0 |
| petal.length | 1.60 | 5.10 | 3.50 | [−3.65, 10.35] | none |
| petal.width | 0.30 | 1.80 | 1.50 | [−1.95, 4.05] | none |

**Why do the petal features show no outliers?** Because the dataset contains two well-separated clusters (Setosa vs the rest). The IQR computed over the pooled data is huge, so the fences are extremely wide and even the extreme values fall inside. This is an important viva point: **the IQR rule is applied per group when the data contains sub-populations** — otherwise real anomalies hide inside the enlarged IQR.

### Violin plots

`violinplot` shows the same grouping plus the density shape. For petal length, the Setosa violin is a thin spike at ~1.5 cm while Virginica is a broader shape around 5–6 cm — a very clear visual demonstration of separability.

---

## 4. Comparisons to write (deliverable 4)

- **Scale:** petal length ranges 1.0–6.9 cm, sepal width only 2.0–4.4 cm. Comparisons should therefore be done within a feature, not across features.
- **Shape:** sepal features are mildly right-skewed (skew ≈ +0.32); petal length is mildly left-skewed (−0.275) because the large Virginica cluster pulls the tail to the left; petal width is nearly symmetric (−0.103).
- **Dispersion:** petal length has the largest standard deviation (1.77), sepal width the smallest (0.44) — petal length varies most across species.
- **Class separation:** `petal.length` and `petal.width` separate Setosa completely and separate Versicolor/Virginica partially. `sepal.width` overlaps almost completely, so it is the weakest feature.
- **Outliers:** only `sepal.width` has IQR outliers in the pooled data — 4 values (2.0, 4.1, 4.2, 4.4). They are genuine biological variation, not measurement errors, so they should not be deleted blindly.
- **Practical implication:** a classifier can reach very high accuracy using only petal measurements; this is exactly what the A6 Naïve Bayes practical demonstrates (only one Versicolor/Virginica error).

---

## 5. Journal questions from the handout (with answers)

**Q1. For the Iris dataset, list the features and their types.**
See the table in section 2: four continuous numeric features (sepal length, sepal width, petal length, petal width, all in cm) plus one nominal categorical feature (`variety`, 3 species, 50 samples each).

**Q2. Write code to create a histogram for each feature (Iris).**
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("iris.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), df.columns[:-1]):
    ax.hist(df[col], bins=15, edgecolor="black")
    ax.set_title(f"Histogram of {col}")
plt.tight_layout(); plt.show()
```

**Q3. Write code to create a box plot for each feature (Iris).**
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), df.columns[:-1]):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Box plot of {col}")
plt.tight_layout(); plt.show()
```

**Q4. Identify the outliers from the box plot of the Iris dataset.**
`sepal.width` has four outliers: 4.4, 4.1, 4.2 (high) and 2.0 (low). The other three features show no points beyond the whiskers in the pooled data. These are valid measurements, so they are retained.

---

## 6. Viva questions the external examiner may ask

**1. What is a histogram?**
A graphical summary of the distribution of a continuous variable: the range is divided into bins and the height of each bar is the count (or density) of observations in that bin. Bars touch because the variable is continuous.

**2. What does the shape of a histogram tell you?**
Whether the distribution is symmetric, skewed left/right, uniform, bimodal, or has gaps/outliers — which guides the choice of summary statistics and transformations.

**3. How do you choose the number of bins?**
There is no single rule: square-root of n, Sturges' rule (`1 + 3.322 log₁₀ n`), or Freedman–Diaconis. Practically, try 10–20 and choose the one that reveals structure without noise.

**4. Difference between a histogram and a bar chart?**
Histogram = continuous numeric data, bins, bars touch, x-axis is a numerical scale. Bar chart = categorical data, separate bars, x-axis is labels.

**5. What is a box plot and what does each part mean?**
Box = Q1 to Q3 (middle 50 %); line = median; whiskers = most extreme values within 1.5 × IQR; dots = outliers. It is the five-number summary.

**6. How are outliers detected in a box plot?**
`value < Q1 − 1.5·IQR` or `value > Q3 + 1.5·IQR`. Extreme outliers use 3 × IQR.

**7. Why does `sepal.width` have outliers while `petal.length` does not?**
Petal length contains two widely separated species clusters, which inflates the pooled IQR and widens the fences. Sepal width's distribution is narrow and overlapping, so four extreme measurements fall outside the fence.

**8. What is the IQR and why is it preferred to the range?**
`Q3 − Q1`, the spread of the middle 50 %. It ignores extreme values, so it is robust; the range only uses two values and is highly sensitive to outliers.

**9. What is skewness? What are the values here?**
Asymmetry of the distribution. Here: sepal length +0.315, sepal width +0.319 (mild right skew), petal length −0.275 (mild left skew), petal width −0.103 (almost symmetric).

**10. What is the difference between a histogram and a KDE?**
A histogram is a discrete binned count; a KDE is a smooth continuous density estimate obtained by summing kernels over the observations. `histplot(kde=True)` shows both.

**11. What is a violin plot? How does it differ from a box plot?**
A violin is a mirrored KDE with an optional inner box. It shows the full distribution shape (bimodality, density) that a box plot hides.

**12. Which Iris features are most useful for classification and why?**
Petal length and petal width — their species distributions are clearly separated (Setosa has petal length ≈ 1–1.9 cm, the others 3–6.9 cm). Sepal width overlaps heavily and is the weakest.

**13. What does `df.describe()` add to the plots?**
Exact numbers: count, mean, std, min, quartiles, max. Plots show shape, statistics quantify it.

**14. How would you compare two distributions formally?**
Visually with overlaid histograms/violins and Q-Q plots; numerically with mean/median/SD/IQR/skew; statistically with a t-test/Mann–Whitney U test or KS test for distribution equality.

**15. What are the alternatives to `pd.read_csv` for the Iris dataset?**
`sklearn.datasets.load_iris()` returns a NumPy array + feature names; `sns.load_dataset("iris")` returns a DataFrame (the column is named `species` there, not `variety`).

**16. What is the difference between `variety` and `species` in Iris?**
Only the column name — same three classes. The repo's `iris.csv` uses `variety`, seaborn's version uses `species`. Code written for one will fail on the other.

**17. Should the outliers in `sepal.width` be removed?**
No. They are plausible biological measurements, not data-entry errors. Removing them would lose genuine information; at most, note their presence or use a robust model.

**18. What is a bimodal distribution? What causes it here?**
A distribution with two peaks. The petal measurements are bimodal because the data mixes two groups of species with different sizes (Setosa vs the other two).

**19. What is data visualisation's role before modelling?**
It reveals distribution shape, outliers, missing patterns and class separability; it guides feature selection, transformation choices and the choice of model.

**20. Why is `petal.width` nearly symmetric but `petal.length` slightly left-skewed?**
Petal width's two clusters are more balanced and closer together, producing a symmetric pooled shape. For petal length the large Virginica cluster lies at high values with a long tail toward the lower values, giving a mild negative skew.

---

## 7. Common mistakes

- Forgetting `plt.tight_layout()` and getting overlapping subplot titles.
- Plotting `variety` on a numeric axis (it must be used with `hue`/`x`).
- Using `df.columns[:-1]` when the target column is not last.
- Reporting outliers for the pooled petal columns and concluding "there are none" without explaining the cluster effect.
- Calling the box-plot whisker "the maximum" when outliers exist.
- Saying the outliers must be removed — state whether they are errors or genuine variation.
- Mixing up seaborn's `species` column with the repo's `variety` column.
