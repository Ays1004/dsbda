# Practical A1 — Data Wrangling-I

**Aim:** Perform data wrangling operations (loading, inspecting, cleaning, formatting, normalising and encoding) on an open-source dataset using Python.

**Dataset:** Iris dataset — 150 rows × 5 columns (4 numeric measurements + 1 categorical species).
**Source:** <https://www.kaggle.com/datasets/uciml/iris> (original UCI source: <https://archive.ics.uci.edu/dataset/53/iris>)
**File used:** `iris.csv`

---

## Full Code

```python
# ============================================================
# Practical A1 : Data Wrangling-I
# Dataset : iris.csv  (150 rows, 5 columns)
# ============================================================

# ---------------- Step 1: Import libraries ----------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------- Step 2 & 3: Load the dataset ----------------
# pd.read_csv() reads a comma separated file into a DataFrame
df = pd.read_csv("iris.csv")

# ---------------- Step 4: Data Preprocessing ----------------

# 4.1 Dimensions of the DataFrame
print("Shape (rows, columns):", df.shape)     # (150, 5)
print("Size (total elements):", df.size)      # 750

# 4.2 First 5 and last 5 records
print("\nFirst 5 records:\n", df.head())
print("\nLast 5 records:\n", df.tail())

# 4.3 Structure of the DataFrame -> columns, dtypes, non-null count, memory
df.info()

# 4.4 Initial statistics of all numeric columns
#     count, mean, std, min, 25%, 50% (median), 75%, max
print(df.describe())

# 4.5 Missing values -> True means the cell is null
print(df.isnull().head())      # full table would be 150 rows; showing first 5
print("\nMissing values per column:\n", df.isnull().sum())

# 4.6 Duplicate records
print("Duplicate rows:", df.duplicated().sum())

# 4.7 Variable descriptions
#   sepal.length  : numeric (continuous, float)  -> length of sepal in cm
#   sepal.width   : numeric (continuous, float)  -> width of sepal in cm
#   petal.length  : numeric (continuous, float)  -> length of petal in cm
#   petal.width   : numeric (continuous, float)  -> width of petal in cm
#   variety       : nominal (categorical, text)  -> species of flower

# ---------------- Step 5: Data Formatting / Type conversion ----------------
print(df.dtypes)

# Convert a column that was read as object into a proper numeric type
df["sepal.length"] = pd.to_numeric(df["sepal.length"], errors="coerce")
df["sepal.length"] = df["sepal.length"].astype("float64")

# Convert the categorical text column into pandas 'category' dtype
df["variety"] = df["variety"].astype("category")
print(df.dtypes)
print(df["variety"].cat.categories)

# ---------------- Step 6: Categorical -> Quantitative ----------------

# (a) Manual mapping. Use this when the categories have an order (ordinal)
df_map = df.copy()
df_map["variety"] = df_map["variety"].astype(str)
df_map["variety"] = df_map["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})
print(df_map.head())

# (b) Label Encoding. Automatically gives 0 to n-1 (alphabetical order)
df_le = df.copy()
df_le["variety"] = df_le["variety"].astype(str)
le = LabelEncoder()
df_le["variety_encoded"] = le.fit_transform(df_le["variety"])
print("Classes seen by LabelEncoder:", list(le.classes_))
print(df_le.head())

# (c) One-Hot Encoding. Use this when the categories have NO order (nominal)
df_ohe = pd.get_dummies(df, columns=["variety"])
print(df_ohe.head())

# ---------------- Step 7: Data Normalisation (Min-Max) ----------------
numeric_cols = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

df_norm = df.copy()
scaler = MinMaxScaler()
df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
print(df_norm.head())

# Same thing done manually: x' = (x - min) / (max - min)
df_norm["petal.width_manual"] = (
    (df["petal.width"] - df["petal.width"].min())
    / (df["petal.width"].max() - df["petal.width"].min())
)
print(df_norm[["petal.width", "petal.width_manual"]].head())

# ---------------- Extra operations from the manual ----------------
# Drop a column (axis=1 -> column, axis=0 -> row)
dropped = df.drop(["petal.length"], axis=1)
print(dropped.head())

# Replace values of a categorical column
df_replaced = df.copy()
df_replaced["variety"] = df_replaced["variety"].astype(str)
df_replaced["variety"] = df_replaced["variety"].replace(
    ["Setosa", "Virginica"], [0, 1]
)
print(df_replaced.head())

# Save the cleaned DataFrame for later practicals
df.to_csv("iris_cleaned.csv", index=False)
```

## Output (abridged)

```text
Shape (rows, columns): (150, 5)
Size (total elements): 750

First 5 records:
   sepal.length  sepal.width  petal.length  petal.width variety
0           5.1          3.5           1.4          0.2  Setosa
1           4.9          3.0           1.4          0.2  Setosa
2           4.7          3.2           1.3          0.2  Setosa
3           4.6          3.1           1.5          0.2  Setosa
4           5.0          3.6           1.4          0.2  Setosa

Last 5 records:
     sepal.length  sepal.width  petal.length  petal.width    variety
145           6.7          3.0           5.2          2.3  Virginica
146           6.3          2.5           5.0          1.9  Virginica
147           6.5          3.0           5.2          2.0  Virginica
148           6.2          3.4           5.4          2.3  Virginica
149           5.9          3.0           5.1          1.8  Virginica

<class 'pandas.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal.length  150 non-null    float64
 1   sepal.width   150 non-null    float64
 2   petal.length  150 non-null    float64
 3   petal.width   150 non-null    float64
 4   variety       150 non-null    object      <-- pandas 3 shows 'str'
dtypes: float64(4), object(1)
memory usage: 6.0+ KB

       sepal.length  sepal.width  petal.length  petal.width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

Missing values per column:
 sepal.length    0
sepal.width     0
petal.length    0
petal.width     0
variety         0
dtype: int64
Duplicate rows: 1
```

After categorical encoding:

```text
Classes seen by LabelEncoder: ['Setosa', 'Versicolor', 'Virginica']

   sepal.length  sepal.width  petal.length  petal.width  variety  variety_encoded
0           5.1          3.5           1.4          0.2   Setosa                0
1           4.9          3.0           1.4          0.2   Setosa                0
2           4.7          3.2           1.3          0.2   Setosa                0
```

After Min-Max scaling:

```text
   sepal.length  sepal.width  petal.length  petal.width variety
0      0.222222     0.625000      0.067797     0.041667  Setosa
1      0.166667     0.416667      0.067797     0.041667  Setosa
2      0.111111     0.500000      0.050847     0.041667  Setosa
```

---

## How to run

```bash
pip install pandas numpy scikit-learn
python a1.py          # or paste the blocks into a Jupyter notebook
```

Keep `iris.csv` in the same folder as the script/notebook.
