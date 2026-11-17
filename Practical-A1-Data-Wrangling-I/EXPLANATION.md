# Practical A1 — Explanation & Viva Questions

## 1. What the practical is actually asking

The examiner wants to see that you can take a raw CSV and answer four things:

1. **Can you load it?** → `pd.read_csv()`
2. **Can you inspect it?** → `shape`, `head()`, `info()`, `describe()`, `isnull()`
3. **Can you fix it?** → type conversion, dropping columns, handling duplicates
4. **Can you prepare it for a model?** → convert text categories to numbers and scale numeric columns

One-line aim: *"To perform data wrangling — inspecting, cleaning, formatting and normalising an open-source dataset (Iris) using pandas."*

---

## 2. About the dataset

| Property | Value |
|---|---|
| Name | Iris dataset |
| Rows | 150 |
| Columns | 5 |
| Numeric features | `sepal.length`, `sepal.width`, `petal.length`, `petal.width` (all `float64`) |
| Categorical feature | `variety` (Setosa / Versicolor / Virginica) — **nominal**, 50 rows each |
| Missing values | None |

**Source to write in the journal:** `https://www.kaggle.com/datasets/uciml/iris` (originally from UCI ML Repository, 1936, R. A. Fisher).

---

## 3. Line-by-line explanation

### Step 1 — Import libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
```

- `pandas` → tabular data (DataFrame, Series) and file I/O.
- `numpy` → fast numerical arrays; pandas is built on top of it.
- `LabelEncoder` → converts text labels into integers 0…n-1.
- `MinMaxScaler` → scales a column to the range [0, 1].

`as` creates an **alias** so we write `pd.read_csv()` instead of `pandas.read_csv()`.

### Step 2 & 3 — Load the data

```python
df = pd.read_csv("iris.csv")
```

`read_csv` reads the header row, infers each column's `dtype`, and returns a **DataFrame**. `df` is just a variable name; nothing is printed until you call `print(df)` or put `df` as the last line of a notebook cell.

### Step 4 — Data preprocessing (inspection)

| Code | What it tells you |
|---|---|
| `df.shape` | Tuple `(rows, columns)` → `(150, 5)` |
| `df.size` | Total cells → `150 × 5 = 750` |
| `df.head()` / `df.tail()` | First/last 5 rows (pass a number for more) |
| `df.info()` | Column names, non-null count, dtype, memory usage |
| `df.describe()` | count, mean, std, min, 25%, 50%, 75%, max for numeric columns |
| `df.isnull()` | Boolean DataFrame — `True` where data is missing |
| `df.isnull().sum()` | Column-wise count of missing values |
| `df.duplicated().sum()` | Number of fully duplicated rows |

Important detail: `describe()` on this file gives `count = 150` for all four numeric columns, which itself proves there are no missing numeric values — no separate check needed.

### Step 5 — Data formatting / type conversion

```python
df["sepal.length"] = pd.to_numeric(df["sepal.length"], errors="coerce")
df["sepal.length"] = df["sepal.length"].astype("float64")
df["variety"] = df["variety"].astype("category")
```

- `pd.to_numeric(..., errors="coerce")` → anything that cannot be parsed becomes `NaN` instead of raising an error.
- `.astype("float64")` → explicit cast.
- Converting `variety` to `category` saves memory and is the "correct" dtype for a nominal variable.

Handout wording "check the data types (character, numeric, integer, factor, logical)" maps to pandas dtypes as:

| Statistical type | pandas dtype |
|---|---|
| Numeric / continuous | `float64`, `int64` |
| Character / string | `object` |
| Factor / categorical | `category` |
| Logical / boolean | `bool` |

### Step 6 — Categorical → quantitative

Three standard ways, and you should be able to say **when** each is used:

1. **Manual mapping** — `df["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})`
   Used when the order matters (small < medium < large). You control the numbers.
2. **Label Encoding** — `LabelEncoder().fit_transform(...)`
   Gives 0…n-1 automatically. Good for **tree-based models**; bad for linear models because it invents a false order (Setosa < Versicolor < Virginica).
3. **One-Hot Encoding** — `pd.get_dummies(df, columns=["variety"])`
   Creates one binary column per category. No false ordering. Use for **nominal** variables with linear/logistic regression.

### Step 7 — Normalisation

```python
x' = (x - min) / (max - min)
```

After Min-Max scaling every value lies in **[0, 1]** and the minimum becomes 0, maximum becomes 1. This is required when features have very different ranges (e.g. age in years vs salary in lakhs) so that a distance-based or gradient-based algorithm is not dominated by the larger column.

> Do **not** confuse this with **standardisation** `(x - μ)/σ`, which gives mean 0 and std 1 but no fixed min/max.

---

## 4. Journal questions from the handout (with answers)

**Q1. Explain data preprocessing in detail.**
Data preprocessing converts raw, noisy data into a clean, model-ready form. Steps: (i) data cleaning — missing values, noise, duplicates, outliers; (ii) data integration — merge multiple sources; (iii) data transformation — normalisation, standardisation, encoding, aggregation; (iv) data reduction — feature selection, dimensionality reduction (PCA); (v) data discretisation — bins instead of continuous values. "Garbage in, garbage out" — a model is only as good as its data.

**Q2. Explain DataFrame with a suitable example.**
A DataFrame is a 2-D, size-mutable, labelled table with rows and columns (like an Excel sheet). Each column is a Series with a single dtype, but different columns may have different dtypes.
```python
import pandas as pd
df = pd.DataFrame({"Name": ["A", "B"], "Marks": [80, 90]})
print(df.shape, df.columns, df.dtypes)
```

**Q3. What is the limitation of label encoding?**
It assigns arbitrary integers, so the model assumes an **order and equal spacing** that may not exist (Setosa=0, Versicolor=1, Virginica=2 implies Virginica > Versicolor > Setosa and that the gap 0→1 equals 1→2). For nominal data this injects false information. It is also sensitive to the alphabetical ordering chosen by the encoder. Fix: use one-hot encoding for nominal features.

**Q4. What is the need of data normalisation?**
To bring all features to a comparable scale so that (a) no feature dominates distance calculations (KNN, K-means, SVM), (b) gradient descent converges faster and more stably, (c) coefficients of regression are comparable, and (d) algorithms assuming a bounded range (neural networks) work properly.

**Q5. Techniques for handling missing data.**
Delete the row/column (`dropna`), mean/median/mode imputation (`fillna`), forward/backward fill (`ffill`, `bfill`), constant placeholder (`"Unknown"`, `-1`), interpolation (`interpolate`), model-based imputation (`KNNImputer`, `IterativeImputer`), or a missing-indicator column. Median is preferred over mean when the column has outliers; mode is used for categorical columns.

---

## 5. Viva questions the external examiner may ask

**1. What is data wrangling?**
The process of cleaning, transforming and organising raw data into a usable format for analysis — includes handling missing values, duplicates, data-type conversion, normalisation and encoding.

**2. Difference between data wrangling and data cleaning?**
Cleaning is one part of wrangling (fixing errors, nulls, duplicates). Wrangling is broader: it also includes merging sources, reshaping, type conversion, scaling and feature creation.

**3. What is a Series vs a DataFrame?**
A Series is a 1-D labelled array (one column). A DataFrame is a 2-D table made of Series sharing the same index.

**4. Difference between `df.shape` and `df.size`?**
`shape` = (rows, columns); `size` = rows × columns (total cells).

**5. What does `df.info()` show?**
Index dtype and range, column names, non-null count per column, dtype of each column, and memory usage. It is the fastest way to spot nulls and wrong dtypes.

**6. What does `df.describe()` return and what is `50%`?**
Count, mean, std, min, quartiles and max for numeric columns. `50%` is the **median** (2nd quartile).

**7. Why is `isnull().sum()` preferred over `isnull()`?**
`isnull()` prints a huge Boolean table; `.sum()` collapses it into one number per column (True = 1), which is what you actually report.

**8. How do you remove duplicates?**
`df.drop_duplicates(inplace=True)`; `df.duplicated().sum()` counts them first.

**9. Difference between `dropna()` and `fillna()`?**
`dropna()` removes rows/columns containing nulls; `fillna(value)` replaces nulls with a given value (mean/median/mode/constant).

**10. What does `inplace=True` do?**
Applies the change to the same object and returns `None`, instead of returning a modified copy. Dangerous in chains — most pandas methods return a new object by default.

**11. What is the difference between `axis=0` and `axis=1`?**
`axis=0` operates down the rows (on each column); `axis=1` operates across the columns (on each row). `df.drop("col", axis=1)` drops a column.

**12. What is the difference between `loc` and `iloc`?**
`loc` selects by **label** (`df.loc[0, "age"]`), `iloc` by **integer position** (`df.iloc[0, 1]`).

**13. Why convert `object` to `category`?**
Saves memory (categories are stored as integer codes) and marks the column semantically as nominal, enabling `.cat` accessors and faster groupby.

**14. Label encoding vs one-hot encoding?**
Label → single integer column, compact, but implies order. One-hot → one binary column per category, no order implied, but increases dimensionality (curse of dimensionality with many categories).

**15. What is the difference between normalisation and standardisation?**
Min-Max normalisation maps to [0,1] using min/max and is sensitive to outliers. Standardisation (Z-score) maps to mean 0, std 1 using mean/σ and is preferred when the data is roughly Gaussian.

**16. Does Min-Max scaling remove outliers?**
No. It only compresses the range; an outlier still sits at 0 or 1 and can squash all other values. Handle/remove outliers first.

**17. What is the difference between nominal and ordinal data?**
Nominal = categories with no order (colour, city, species). Ordinal = categories with a meaningful order (small/medium/large, grades A/B/C).

**18. What is a data dictionary / variable description?**
A table describing every column: name, meaning, unit, data type, allowed range. The handout explicitly asks for it, so write it in the journal.

**19. Why do we need to encode categorical variables at all?**
ML algorithms work on numbers and compute distances/weights. Text cannot be multiplied by a coefficient, so it must be converted to a numeric representation.

**20. What is the "curse of dimensionality" in the context of one-hot encoding?**
A column with 1000 unique categories becomes 1000 new columns, making the data sparse and models slow and prone to overfitting. Use `pd.get_dummies(drop_first=True)` or target/frequency encoding.

---

## 6. Common mistakes (avoid in the practical)

- Calling `describe()` and `info()` without `print()` in a **script** → no output (works in Jupyter because of the last-expression rule).
- Using `inplace=True` inside an assignment: `df = df.fillna(0, inplace=True)` makes `df` become `None`.
- Encoding the target column of the test set separately with `fit_transform` instead of `transform` → inconsistent codes.
- Forgetting `index=False` in `to_csv` → writes an extra unnamed index column.
- Saying normalisation "removes outliers" — it does not.
