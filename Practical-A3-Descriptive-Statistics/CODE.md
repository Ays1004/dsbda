# Practical A3 — Descriptive Statistics

**Aim:** Compute measures of central tendency (mean, median, mode) and variability (min, max, standard deviation, percentiles) for numeric variables grouped by a qualitative variable, using pandas.

**Datasets:**
1. A small `age` / `income` / `age_group` dataset created in the program (handout part 1).
2. `iris.csv` — Iris dataset, grouped by `variety` (handout part 2).

**Libraries:** `pandas`

---

## Full Code

```python
# ============================================================
# Practical A3 : Descriptive Statistics
# Measures of central tendency and variability
# ============================================================

import pandas as pd

# ============================================================
# PART 1 : income grouped by age group  (synthetic data.csv)
# ============================================================

data = {
    "age":      [25, 30, 22, 40, 55, 60, 33, 28, 45, 50],
    "income":   [50000, 60000, 45000, 70000, 80000, 90000, 65000, 62000, 75000, 85000],
    "age_group": ["20-30", "30-40", "20-30", "40-50", "50-60",
                  "50-60", "30-40", "20-30", "40-50", "50-60"],
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# --- Summary statistics of income grouped by age_group ---
# describe() gives count, mean, std, min, 25%, 50%, 75%, max
summary_stats = df.groupby("age_group")["income"].describe()
print("\nSummary statistics of income by age group:\n", summary_stats)

# --- Median income per group ---
median_income = df.groupby("age_group")["income"].median()
print("\nMedian income by age group:\n", median_income)

# --- Mean, min, max, std per group, one by one ---
print("\nMean income:\n", df.groupby("age_group")["income"].mean())
print("\nMinimum income:\n", df.groupby("age_group")["income"].min())
print("\nMaximum income:\n", df.groupby("age_group")["income"].max())
print("\nStandard deviation of income:\n", df.groupby("age_group")["income"].std())

# --- Column names ---
print("\nColumn names:", df.columns)

# --- A list containing a numeric value for each response ---
income_list = df["income"].tolist()
print("\nIncome values as a list:", income_list)

# ============================================================
# PART 1(b) : mode  (data with repeated values)
# ============================================================

data2 = {
    "age":      [25, 30, 25, 40, 55, 60, 33, 28, 45, 50, 25, 30, 28, 30, 25],
    "income":   [50000, 60000, 50000, 70000, 80000, 90000, 65000, 62000,
                 75000, 85000, 50000, 60000, 62000, 70000, 75000],
    "age_group": ["20-30", "30-40", "20-30", "40-50", "50-60", "50-60",
                  "30-40", "20-30", "40-50", "50-60", "20-30", "30-40",
                  "20-30", "30-40", "20-30"],
}
df2 = pd.DataFrame(data2)

print("\nMode of age:   ", df2["age"].mode().values)
print("Mode of income:", df2["income"].mode().values)

# ============================================================
# PART 2 : iris.csv grouped by variety (species)
# ============================================================

iris = pd.read_csv("iris.csv")
print("\nFirst 5 rows of iris:\n", iris.head())

# --- Summary statistics of every numeric column, per species ---
print("\nSummary statistics by species:\n", iris.groupby("variety").describe())

# --- Percentiles (25th, 50th, 75th) per species ---
percentiles = iris.groupby("variety").quantile([0.25, 0.5, 0.75])
print("\nPercentiles by species:\n", percentiles)

# --- Sepal width statistics per species ---
print("\nSepal width statistics by species:\n",
      iris.groupby("variety")["sepal.width"].describe())

# --- Median of all numeric columns per species ---
print("\nMedian values by species:\n", iris.groupby("variety").median(numeric_only=True))

# --- Median sepal length per species ---
print("\nMedian sepal length by species:\n",
      iris.groupby("variety")["sepal.length"].median())

# --- Mode of sepal width ---
print("\nMode of sepal width:", iris["sepal.width"].mode().values)

# --- Overall statistics (not grouped) ---
print("\nOverall description:\n", iris.describe())

# --- Mean and standard deviation for each species (handout wording) ---
for sp in ["Setosa", "Versicolor", "Virginica"]:
    subset = iris[iris["variety"] == sp]
    print(f"\n{sp}:")
    print(f"  sepal.length -> mean = {subset['sepal.length'].mean():.3f}, "
          f"std = {subset['sepal.length'].std():.3f}, "
          f"min = {subset['sepal.length'].min()}, "
          f"max = {subset['sepal.length'].max()}")
```

## Output (abridged)

```text
Summary statistics of income by age group:
            count          mean          std   ...      50%      75%      max
age_group                                     ...
20-30        3.0  52333.333333  8736.894948   ...  50000.0  56000.0  62000.0
30-40        2.0  62500.000000  3535.533906   ...  62500.0  63750.0  65000.0
40-50        2.0  72500.000000  3535.533906   ...  72500.0  73750.0  75000.0
50-60        3.0  85000.000000  5000.000000   ...  85000.0  87500.0  90000.0

Median income by age group:
 age_group
20-30    50000.0
30-40    62500.0
40-50    72500.0
50-60    85000.0

Column names: Index(['age', 'income', 'age_group'], dtype='object')

Mode of age:    [25]
Mode of income: [50000]

Percentiles by species:
                  sepal.length  sepal.width  petal.length  petal.width
variety
Setosa     0.25         4.800        3.200         1.400          0.2
           0.50         5.000        3.400         1.500          0.2
           0.75         5.200        3.675         1.575          0.3
Versicolor 0.25         5.600        2.525         4.000          1.2
           0.50         5.900        2.800         4.350          1.3
           0.75         6.300        3.000         4.600          1.5
Virginica  0.25         6.225        2.800         5.100          1.8
           0.50         6.500        3.000         5.550          2.0
           0.75         6.900        3.175         5.875          2.3

Median values by species:
             sepal.length  sepal.width  petal.length  petal.width
variety
Setosa               5.0          3.4          1.50          0.2
Versicolor           5.9          2.8          4.35          1.3
Virginica            6.5          3.0          5.55          2.0

Mode of sepal width: [3.]

Setosa:
  sepal.length -> mean = 5.006, std = 0.352, min = 4.3, max = 5.8
Versicolor:
  sepal.length -> mean = 5.936, std = 0.516, min = 4.9, max = 7.0
Virginica:
  sepal.length -> mean = 6.588, std = 0.636, min = 4.9, max = 7.9
```

---

## One-line definitions to remember

| Measure | pandas call | Meaning |
|---|---|---|
| Mean | `.mean()` | arithmetic average, affected by outliers |
| Median | `.median()` | middle value after sorting, robust |
| Mode | `.mode()` | most frequent value (can be multiple) |
| Standard deviation | `.std()` | spread around the mean (sample, `ddof=1`) |
| Variance | `.var()` | square of standard deviation |
| Percentile | `.quantile(0.25)` | value below which 25 % of observations fall |
| Five-number summary | `.describe()` | count, mean, std, min, Q1, Q2, Q3, max |
