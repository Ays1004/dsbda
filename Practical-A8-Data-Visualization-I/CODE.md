# Practical A8 — Data Visualization-I

**Aim:** Use Seaborn on the built-in `titanic` dataset to find patterns in the data, and plot the distribution of the ticket `fare` using a histogram.

**Dataset:** `titanic` — 891 passengers, 15 columns, loaded directly from Seaborn.
**Libraries:** `seaborn`, `matplotlib`, `pandas`

> Requires internet the first time (`sns.load_dataset` downloads and caches the CSV in `~/seaborn-data`). Offline fallback is given below.

---

## Full Code

```python
# ============================================================
# Practical A8 : Data Visualization-I (Titanic dataset)
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- 1. Load the built-in dataset ----------------
df = sns.load_dataset("titanic")
# Offline alternative:
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
# df = pd.read_csv(url)

print("Shape:", df.shape)          # (891, 15)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDescription:\n", df.describe())

# ---------------- 2. Look for patterns (numbers first, plots after) ----------------
print("\nOverall survival rate:", round(df["survived"].mean(), 4))
print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:\n", df.groupby("class")["survived"].mean())
print("\nSurvival rate by sex and class:\n",
      df.groupby(["sex", "class"])["survived"].mean())
print("\nPassengers by sex:\n", df["sex"].value_counts())
print("\nPassengers by class:\n", df["class"].value_counts())

# ---------------- 3. Distribution of age ----------------
plt.figure(figsize=(6, 4))
sns.histplot(df["age"].dropna(), bins=20, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Number of passengers")
plt.show()

# ---------------- 4. REQUIRED: distribution of fare (histogram) ----------------
plt.figure(figsize=(7, 4))
sns.histplot(x="fare", bins=10, data=df)
plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# Fare histogram split by survival
plt.figure(figsize=(7, 4))
sns.histplot(x="fare", bins=10, data=df, hue="survived", multiple="stack")
plt.title("Fare Distribution by Survival")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# ---------------- 5. Category vs numeric plots ----------------
# Box plot: age distribution per class
plt.figure(figsize=(6, 4))
sns.boxplot(x="class", y="age", data=df, palette="pastel")
plt.title("Age distribution by passenger class")
plt.show()

# Violin plot: age distribution per class
plt.figure(figsize=(6, 4))
sns.violinplot(x="class", y="age", data=df, palette="rainbow")
plt.title("Age distribution by class (violin)")
plt.show()

# Count plot: survivors by sex
plt.figure(figsize=(6, 4))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival count by sex")
plt.show()

# Scatter plot: age vs fare, coloured by survival
plt.figure(figsize=(6, 4))
sns.scatterplot(x="age", y="fare", hue="survived", data=df)
plt.title("Age vs Fare by survival")
plt.show()

# ---------------- 6. Correlation heat map ----------------
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ---------------- 7. Pair plot (multivariate pattern finding) ----------------
sns.pairplot(df[["age", "fare", "pclass", "survived"]].dropna(), hue="survived")
plt.show()
```

## Output (abridged)

```text
Shape: (891, 15)
Columns: ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
          'embarked', 'class', 'who', 'adult_male', 'deck',
          'embark_town', 'alive', 'alone']

Missing values per column:
age            177
deck           688
embarked         2
(rest             0)

Overall survival rate: 0.3838

Survival rate by sex:
female    0.742038
male      0.188908

Survival rate by class:
First     0.629630
Second    0.472826
Third     0.242363

Survival rate by sex and class:
sex     class
female  First     0.968085
        Second    0.921053
        Third     0.500000
male    First     0.368852
        Second    0.157407
        Third     0.135447

Passengers by sex:   male 577, female 314
Passengers by class: Third 491, First 216, Second 184
```

The plotting calls open interactive windows; in a notebook they appear inline. The important reading is from the fare histogram:

```text
count    891.000000
mean      32.204208
std       49.693429
min        0.000000
25%        7.910400
50%       14.454200
75%       31.000000
max      512.329200
```

Mean (32.2) is more than twice the median (14.45) and the maximum is 512 — a clear **right-skewed** distribution.

---

## Observations to write in the journal

1. **Fare is heavily right-skewed.** Most passengers paid less than 50, a few paid more than 500 — a long tail of first-class tickets. Mean fare (≈ 32) is much larger than the median (≈ 14).
2. **Survival is not random with respect to sex.** Female survival ≈ 74 %, male survival ≈ 19 % — "women and children first" is clearly visible.
3. **Class matters.** 1st class survival ≈ 63 %, 2nd ≈ 47 %, 3rd ≈ 24 %.
4. **Age is roughly normal** but has a spike for infants and some missing values (177 nulls).
5. `fare` is positively correlated with `pclass`/`survived`, which is expected because 1st-class tickets cost more and those passengers survived more often.

---

## When to use which plot

| Plot | Use |
|---|---|
| `histplot` / `displot` | distribution of one numeric variable |
| `kdeplot` | smooth density estimate |
| `boxplot` | five-number summary, spotting outliers, numeric vs category |
| `violinplot` | box plot + density shape, numeric vs category |
| `countplot` | frequency of a categorical variable |
| `barplot` | mean of a numeric variable per category (with confidence interval) |
| `scatterplot` | relationship between two numeric variables |
| `pairplot` | pairwise relationships of many numeric variables |
| `heatmap` | correlation matrix / matrix data |
| `catplot` | figure-level interface to any categorical plot |
