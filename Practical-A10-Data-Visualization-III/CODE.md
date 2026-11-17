# Practical A10 — Data Visualization-III

**Aim:** Load the Iris dataset into a DataFrame, list its features and types, plot a histogram and a box plot for each feature, compare distributions and identify outliers.

**Dataset:** Iris — 150 rows × 5 columns.
**Source:** <https://archive.ics.uci.edu/dataset/53/iris>
**File used:** `iris.csv`

**Libraries:** `pandas`, `matplotlib`, `seaborn`

---

## Full Code

```python
# ============================================================
# Practical A10 : Data Visualization-III (Iris dataset)
# Histograms + box plots for every feature, outlier detection
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 1. Load the dataset ----------------
df = pd.read_csv("iris.csv")
print("Shape:", df.shape)          # (150, 5)
print("\nFirst 5 rows:\n", df.head())

# ---------------- 2. Features and their types ----------------
print("\nFeatures and their data types:")
print(df.dtypes)

print("\nFeature description:")
print("""
sepal.length : numeric, continuous (float) - sepal length in cm
sepal.width  : numeric, continuous (float) - sepal width in cm
petal.length : numeric, continuous (float) - petal length in cm
petal.width  : numeric, continuous (float) - petal width in cm
variety      : nominal, categorical (text) - Setosa / Versicolor / Virginica
""")

print("Number of samples per species:\n", df["variety"].value_counts())

features = df.columns[:-1].tolist()      # all numeric columns
print("\nNumeric features:", features)

# ---------------- 3. Histogram for each feature ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), features):
    ax.hist(df[col], bins=15, edgecolor="black", color="skyblue")
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.suptitle("Iris - Histograms of all features", fontsize=14)
plt.tight_layout()
plt.show()

# Histograms separated by species (shows why the classes are separable)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(data=df, x=col, hue="variety", bins=15, ax=ax, element="step")
    ax.set_title(f"{col} by species")
plt.tight_layout()
plt.show()

# ---------------- 4. Box plot for each feature ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(y=df[col], ax=ax, color="salmon")
    ax.set_title(f"Box plot of {col}")
plt.suptitle("Iris - Box plots of all features", fontsize=14)
plt.tight_layout()
plt.show()

# Box plot of each feature grouped by species
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="variety", y=col, ax=ax,
                hue="variety", palette="Set2", legend=False)
    ax.set_title(f"{col} by species")
plt.tight_layout()
plt.show()

# ---------------- 5. Descriptive statistics + comparison ----------------
print("\nSummary statistics:\n", df.describe())
print("\nMean / median / std / skew per feature:\n",
      df[features].agg(["mean", "median", "std", "skew"]).round(3))

# ---------------- 6. Outlier detection with the IQR rule ----------------
print("\n----- Outlier report (IQR method) -----")
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)][col]

    print(f"\nFeature: {col}")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Accepted range: [{lower:.2f}, {upper:.2f}]")
    print(f"  Mean={df[col].mean():.2f}, Median={df[col].median():.2f}, "
          f"Std={df[col].std():.2f}")
    if outliers.empty:
        print("  Outliers: None")
    else:
        print(f"  Outliers ({len(outliers)}): {outliers.tolist()}")

# ---------------- 7. Violin plot (distribution shape per species) ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), features):
    sns.violinplot(data=df, x="variety", y=col, hue="variety",
                   palette="Set2", legend=False, ax=ax)
    ax.set_title(f"Violin plot of {col} by species")
plt.tight_layout()
plt.show()
```

## Output (abridged)

```text
Shape: (150, 5)

Features and their data types:
sepal.length    float64
sepal.width     float64
petal.length    float64
petal.width     float64
variety          object      <-- pandas 3 shows 'str'
dtype: object

Number of samples per species:
Setosa        50
Versicolor    50
Virginica     50

Summary statistics:
       sepal.length  sepal.width  petal.length  petal.width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

Mean / median / std / skew per feature:
         sepal.length  sepal.width  petal.length  petal.width
mean           5.843        3.057         3.758        1.199
median         5.800        3.000         4.350        1.300
std            0.828        0.436         1.765        0.762
skew           0.315        0.319        -0.275       -0.103

----- Outlier report (IQR method) -----
Feature: sepal.length
  Q1=5.10, Q3=6.40, IQR=1.30
  Accepted range: [3.15, 8.35]
  Outliers: None

Feature: sepal.width
  Q1=2.80, Q3=3.30, IQR=0.50
  Accepted range: [2.05, 4.05]
  Outliers (4): [4.4, 4.1, 4.2, 2.0]

Feature: petal.length
  Q1=1.60, Q3=5.10, IQR=3.50
  Accepted range: [-3.65, 10.35]
  Outliers: None

Feature: petal.width
  Q1=0.30, Q3=1.80, IQR=1.50
  Accepted range: [-1.95, 4.05]
  Outliers: None
```

---

## Inferences to write in the journal

1. **Features and types:** four numeric continuous features (`sepal.length`, `sepal.width`, `petal.length`, `petal.width`) and one nominal categorical feature (`variety` with 3 values, 50 samples each).
2. **Petal measurements are strongly bimodal** in the overall histogram — one cluster around 1–2 cm (Setosa) and another around 3–7 cm (Versicolor/Virginica). Sepal measurements overlap much more.
3. **Petal length and petal width separate the species almost perfectly**; sepal width is the weakest discriminator.
4. **Outliers (IQR rule):** `sepal.width` has 4 outliers — three unusually wide sepals (4.1, 4.2, 4.4 cm) and one unusually narrow sepal (2.0 cm). `sepal.length`, `petal.length` and `petal.width` show no IQR outliers, because the gap between species widens the IQR.
5. **Skew:** `sepal.length` (+0.315) and `sepal.width` (+0.319) are mildly right-skewed; `petal.length` (−0.275) is mildly left-skewed; `petal.width` (−0.103) is almost symmetric.
6. **Conclusion:** Iris is close to linearly separable for Setosa; Versicolor and Virginica overlap slightly, which is why any classifier will make a small number of errors between those two classes.
