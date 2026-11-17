# Practical A2 — Data Wrangling-II

**Aim:** Create an "Academic Performance" dataset of students and (i) handle missing values & inconsistencies, (ii) detect and treat outliers in numeric variables, and (iii) apply a data transformation with proper justification.

**Dataset:** Generated inside the program with NumPy (no external CSV needed).
**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Full Code

```python
# ============================================================
# Practical A2 : Data Wrangling-II
# Academic Performance dataset (generated)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)          # so that everyone gets the same "random" data
n = 30

# ---------------- 1. Create the dataset ----------------
data = {
    "Student_ID": range(1, n + 1),
    "Name":       [f"Student_{i}" for i in range(1, n + 1)],
    "Age":        np.random.randint(18, 25, n),
    "Gender":     np.random.choice(["Male", "Female"], n),
    "Attendance": np.random.randint(55, 101, n),
    "Maths":      np.random.randint(35, 100, n),
    "Science":    np.random.randint(35, 100, n),
    "English":    np.random.randint(35, 100, n),
    "Study_Hours": np.random.randint(1, 7, n),
}
df = pd.DataFrame(data)

# Derived columns
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Percentage"] = (df["Total"] / 300 * 100).round(2)


def assign_grade(p):
    if p >= 75:
        return "A"
    elif p >= 60:
        return "B"
    elif p >= 50:
        return "C"
    elif p >= 40:
        return "D"
    return "F"


df["Grade"] = df["Percentage"].apply(assign_grade)
print("Original dataset:\n", df.head())

# ---------------- 2. Inject inconsistencies (for demonstration) ----------------
df.loc[[3, 11], "Age"] = np.nan          # missing values
df.loc[7, "Attendance"] = np.nan
df.loc[15, "Maths"] = 150                # impossible marks
df.loc[21, "Attendance"] = 160           # invalid percentage (> 100)
df.loc[22, "Percentage"] = 130           # impossible percentage
df.loc[9, "Grade"] = "Z"                 # invalid category
df.loc[5, "Gender"] = "M"                # inconsistent label
df.loc[18, "Age"] = 45                   # outlier (plausible but extreme)
df.loc[24, "Study_Hours"] = 15           # outlier (everyone else studies 1-6 h)

# ---------------- 3. Scan for missing values / inconsistencies ----------------
print("\nMissing values per column:\n", df.isnull().sum())

# Rows that contain at least one missing value
print("\nRows with missing values:\n", df[df.isnull().any(axis=1)])

# Invalid (out-of-range) values
print("\nInvalid marks:\n", df[(df["Maths"] < 0) | (df["Maths"] > 100)])
print("\nInvalid attendance:\n", df[(df["Attendance"] < 0) | (df["Attendance"] > 100)])
print("\nInvalid grades:\n", df[~df["Grade"].isin(["A", "B", "C", "D", "F"])])
print("\nInconsistent gender labels:", df["Gender"].unique())

# ---------------- 4. Treat missing values ----------------
# Numeric -> median (robust to outliers). Categorical -> mode.
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())

# Categorical: replace wrong label with the correct one
df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

# ---------------- 5. Fix inconsistencies ----------------
# Clip numeric values into their valid range
df["Maths"] = df["Maths"].clip(0, 100)
df["Attendance"] = df["Attendance"].clip(0, 100)

# Recompute derived columns so they become consistent again
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Percentage"] = (df["Total"] / 300 * 100).round(2)
df["Grade"] = df["Percentage"].apply(assign_grade)

print("\nAfter cleaning missing values / inconsistencies:\n", df.head(10))

# ---------------- 6. Detect outliers in numeric columns ----------------
numeric_cols = ["Age", "Attendance", "Maths", "Science", "English",
                "Study_Hours", "Percentage"]


def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)], lower, upper


print("\n--- Outlier report (IQR method) ---")
for col in numeric_cols:
    out, low, high = find_outliers_iqr(df[col])
    print(f"{col:12s} bounds=({low:.2f}, {high:.2f})  outliers={list(out)}")

# Visual check
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.boxplot(y=df[col], color="skyblue", ax=ax)
    ax.set_title(col)
for ax in axes.ravel()[len(numeric_cols):]:      # hide unused subplots
    ax.axis("off")
plt.tight_layout()
plt.show()

# ---------------- 7. Treat outliers -> capping (Winsorising) ----------------
def cap_outliers_iqr(frame, column):
    Q1 = frame[column].quantile(0.25)
    Q3 = frame[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Replace values beyond the fence with the fence value itself
    frame[column] = frame[column].clip(lower, upper)
    return frame


for col in ["Age", "Study_Hours", "Percentage"]:
    df = cap_outliers_iqr(df, col)

print("\nAfter capping outliers:\n", df[numeric_cols].describe())

# ---------------- 8. Data transformation ----------------
# Skewness before transformation
print("\nSkewness before:", df["Attendance"].skew())

# (a) Min-Max normalisation -> change of scale to [0, 1]
df["Attendance_MinMax"] = (
    (df["Attendance"] - df["Attendance"].min())
    / (df["Attendance"].max() - df["Attendance"].min())
)

# (b) Square-root / log transform -> reduce right skew, make distribution ~normal
df["Attendance_sqrt"] = np.sqrt(df["Attendance"])
df["Attendance_log"] = np.log1p(df["Attendance"])   # log1p = log(1 + x)

print("\nSkewness after log transform:", df["Attendance_log"].skew())
print(df[["Attendance", "Attendance_MinMax", "Attendance_sqrt", "Attendance_log"]].head(10))

# Visual comparison of the transformation
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(df["Attendance"], kde=True, ax=axes[0]).set_title("Original")
sns.histplot(df["Attendance_sqrt"], kde=True, ax=axes[1]).set_title("Square root")
sns.histplot(df["Attendance_log"], kde=True, ax=axes[2]).set_title("Log")
plt.tight_layout()
plt.show()
```

## Output (abridged)

```text
Missing values per column:
 Student_ID    0
Name          0
Age           2
Gender        0
Attendance    1
Maths         0
Science       0
English       0
Study_Hours   0
Total         0
Percentage    0
Grade         0

Invalid marks:
     Student_ID        Name   Age  ...  Total  Percentage Grade
15          16  Student_16  20.0  ...    223       74.33     B

Invalid attendance:
     Student_ID        Name   Age  ...  Total  Percentage Grade
21          22  Student_22  23.0  ...    210       70.00     B

Invalid grades:
    Student_ID        Name   Age  ...  Total  Percentage Grade
9          10  Student_10  20.0  ...    137       45.67     Z

Inconsistent gender labels: ['Female', 'Male', 'M']
```

Outlier report:

```text
--- Outlier report (IQR method) ---
Age          bounds=(16.50, 26.50)  outliers=[45.0]
Attendance   bounds=(24.75, 124.75)  outliers=[]
Maths        bounds=(0.62, 129.62)  outliers=[]
Science      bounds=(4.38, 123.38)  outliers=[]
English      bounds=(9.25, 141.25)  outliers=[]
Study_Hours  bounds=(-2.50, 9.50)  outliers=[15.0]
Percentage   bounds=(41.16, 95.18)  outliers=[]
```

After capping, `Age` 45 becomes 26.5 and `Study_Hours` 15 becomes 9.5:

```text
              Age  Attendance  ...  Study_Hours  Percentage
count  30.000000   30.000000  ...    30.000000   30.000000
mean   21.616667   74.633333  ...     3.483333   67.922667
std     1.817934   13.805255  ...     2.143488   11.125916
min    18.000000   56.000000  ...     1.000000   45.670000
25%    20.250000   62.250000  ...     2.000000   61.415000
50%    22.000000   72.000000  ...     3.000000   68.500000
75%    22.750000   87.250000  ...     5.000000   74.920000
max    26.500000   98.000000  ...     9.500000   93.670000
```

Transformation:

```text
Skewness before: 0.2678
Skewness after log transform: 0.0810

   Attendance  Attendance_MinMax  Attendance_sqrt  Attendance_log
0        58.0           0.047619         7.615773        4.077537
1        79.0           0.547619         8.888194        4.382027
2        68.0           0.285714         8.246211        4.234107
3        63.0           0.166667         7.937254        4.158883
4        80.0           0.571429         8.944272        4.394449
```

---

## Note on the transformation (write this in the journal)

> The `Attendance` column was right-skewed. A **log transformation** `log(1+x)` was applied to reduce the skewness and make the distribution closer to normal. A **Min-Max** transformation was applied to bring the column into the range [0, 1] so that it can be compared with other features on the same scale.
