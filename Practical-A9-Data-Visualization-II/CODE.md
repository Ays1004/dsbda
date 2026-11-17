# Practical A9 — Data Visualization-II

**Aim:** Using the built-in `titanic` dataset, plot a **box plot of `age` for each `sex`, split by whether the passenger survived**, and write observations from the plot.

**Dataset:** `titanic` — 891 rows (Seaborn built-in).
**Libraries:** `seaborn`, `matplotlib`, `pandas`

---

## Full Code

```python
# ============================================================
# Practical A9 : Data Visualization-II
# Box plot: age distribution by sex and survival status
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- 1. Load the dataset ----------------
df = sns.load_dataset("titanic")
# Offline alternative:
# df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

print("Shape:", df.shape)
print(df.head())

# ---------------- 2. Overall description ----------------
print("\nDescribe (transposed):\n", df.describe().transpose())

# ---------------- 3. Mean / median / mode of age (manual calculation) ----------------
age_data = df["age"].dropna()
sorted_age = sorted(age_data)
n = len(sorted_age)

mean_age = sum(age_data) / len(age_data)

if n % 2 == 1:                                  # odd number of values
    median_age = sorted_age[n // 2]
else:                                           # even number of values
    median_age = (sorted_age[n // 2 - 1] + sorted_age[n // 2]) / 2

age_counts = Counter(age_data)
mode_age = age_counts.most_common(1)[0][0]

print(f"\nMean age   : {mean_age:.2f}")
print(f"Median age : {median_age:.2f}")
print(f"Mode age   : {mode_age:.2f}")

# pandas one-liners give the same result
print("pandas -> mean:", round(df['age'].mean(), 2),
      " median:", df['age'].median(), " mode:", df['age'].mode().values)

# ---------------- 4. Grouped statistics ----------------
print("\nAge statistics by sex:\n",
      df.groupby("sex")["age"].describe())

print("\nAge statistics by sex and survival:\n",
      df.groupby(["sex", "survived"])["age"].describe())

print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean())

# ---------------- 5. REQUIRED: box plot ----------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="sex", y="age", hue="survived", data=df, palette="viridis")
plt.title("Distribution of Age by Gender and Survival Status")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.legend(title="Survived", labels=["No (0)", "Yes (1)"])
plt.show()

# ---------------- 6. Supporting plots ----------------
# Violin plot - same grouping, shows the density shape
plt.figure(figsize=(8, 5))
sns.violinplot(x="sex", y="age", hue="survived", data=df,
               palette="muted", split=True)
plt.title("Age distribution by sex and survival (violin)")
plt.show()

# Count plot - how many passengers are in each group
sns.catplot(x="sex", hue="survived", data=df, kind="count")
plt.title("Passenger count by sex and survival")
plt.show()

# Box plot of fare by class, for comparison
plt.figure(figsize=(7, 4))
sns.boxplot(x="class", y="fare", data=df, palette="Set2")
plt.title("Fare distribution by passenger class")
plt.show()
```

## Output (abridged)

```text
Mean age   : 29.70
Median age : 28.00
Mode age   : 24.00
pandas -> mean: 29.7   median: 28.0   mode: [24.]

Age statistics by sex:
         count       mean        std   min   25%   50%   75%   max
sex
female   261.0  27.915709  14.110146  0.75  18.0  27.0  37.0  63.0
male     453.0  30.726645  14.678201  0.42  21.0  29.0  39.0  80.0

Age statistics by sex and survival:
                  count       mean        std   min    25%   50%    75%   max
sex    survived
female 0          64.0  25.046875  13.618591  2.00  16.75  24.5  33.25  57.0
       1         197.0  28.847716  14.175073  0.75  19.00  28.0  38.00  63.0
male   0         360.0  31.618056  14.056019  1.00  21.75  29.0  39.25  74.0
       1          93.0  27.276022  16.504803  0.42  18.00  28.0  36.00  80.0

Survival rate by sex:
female    0.742038
male      0.188908
```

Four boxes are produced by the required call (female/0, female/1, male/0, male/1). The plotted output should be pasted into the journal, with the observations listed in the next section.

---

## Observations to write in the journal (the handout asks for this)

1. **Age alone does not separate survivors from non-survivors strongly.** The boxes overlap heavily. For females the median age is 24.5 for non-survivors and 28 for survivors; for males it is 29 (died) vs 28 (survived).
2. **Sex is the dominant factor.** Female survival ≈ 74 %, male ≈ 19 %. The gap is visible in the survival rates even though the age boxes look similar.
3. **Children were more likely to survive**, seen as the lower whisker/points around 0–10 years in the "survived = 1" group — consistent with "women and children first". The youngest survivor is 0.42 years old.
4. **Age has missing values** (177 of 891), so each box is drawn from fewer than 891 passengers.
5. **Outliers:** the male groups have several very old passengers (up to 74 for non-survivors and 80 for survivors) shown as individual points; the female groups extend to 57–63.
6. The 1st quartile of both sexes is around 18–22 and the 3rd quartile around 33–39, so the **middle 50 % of passengers were young adults**.

**Inference to state:** "Age by itself is a weak predictor of survival on the Titanic, whereas sex and passenger class are strong predictors. The box plot makes the weak age effect visible and simultaneously exposes the age outliers and missing-data limitation."
