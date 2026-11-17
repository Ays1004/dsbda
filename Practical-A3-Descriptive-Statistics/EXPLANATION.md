# Practical A3 — Explanation & Viva Questions

## 1. What the practical is asking

Two parts, both about **descriptive statistics**:

1. Take numeric variables (`income`) and summarise them **grouped by** a categorical variable (`age_group`). Report mean, median, min, max, standard deviation, percentiles.
2. Do the same for `iris.csv`, grouped by species — mean, standard deviation, percentile for `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`.

The core pandas pattern is always:

```python
df.groupby("categorical_column")["numeric_column"].statistic()
```

You must also be able to *define* mean, median, mode, variance, standard deviation and percentile, and explain **when each is appropriate**.

---

## 2. Line-by-line explanation

### `df.groupby("age_group")["income"]`

`groupby` splits the DataFrame into sub-tables — one per unique value of `age_group`. Selecting `["income"]` keeps only that column. Nothing is computed yet; you then attach an **aggregation**:

| Aggregation | Returns |
|---|---|
| `.mean()` | mean income per group |
| `.median()` | median per group |
| `.min()` / `.max()` | extremes per group |
| `.std()` | sample standard deviation per group |
| `.describe()` | all of the above + count + quartiles at once |
| `.quantile([0.25, 0.5, 0.75])` | chosen percentiles per group |
| `.agg(["mean", "median", "std"])` | several statistics in one table |

This split–apply–combine is exactly what MapReduce does at scale, which is a nice link to Group B.

### `describe()`

Output columns: `count, mean, std, min, 25%, 50%, 75%, max`.
- `count` = number of non-null values (compare with `df.shape[0]` to detect missing data).
- `50%` = median = 2nd quartile.
- `25%` and `75%` = Q1, Q3; `Q3 − Q1` = IQR.
- `std` in pandas is the **sample** standard deviation, denominator `n − 1` (`ddof=1`). NumPy's `np.std` uses `n` by default — a classic viva trap.

### `mode()`

Returns a **Series**, because a column can be multi-modal (two equally frequent values). That is why the code writes `.mode().values` or `.mode()[0]`.

For the modified dataset, age 25 appears 4 times and income 50000 appears 3 times, so mode(age) = 25 and mode(income) = 50000.

### Percentiles

`df.quantile(0.25)` → the value below which 25 % of the observations lie. The median is the 50th percentile; Q1/Q3 are the 25th/75th. Percentiles are **robust** — they do not change much when a few extreme values are added, unlike the mean.

### Grouped loop at the end

```python
for sp in ["Setosa", "Versicolor", "Virginica"]:
    subset = iris[iris["variety"] == sp]
```

`iris["variety"] == sp` produces a Boolean Series; using it inside `iris[...]` keeps only the matching rows. This is **Boolean indexing**, the same as SQL's `WHERE`. It is shown separately because the handout specifically names the three species.

---

## 3. The statistics you must be able to define

| Measure | Formula | Property |
|---|---|---|
| Mean | `Σx / n` | uses all values; affected by outliers |
| Median | middle value of sorted data | robust; equals 50th percentile |
| Mode | most frequent value | only measure usable for nominal data |
| Range | `max − min` | crude, sensitive to extremes |
| Variance (sample) | `Σ(x − x̄)² / (n − 1)` | squared units |
| Standard deviation | `√variance` | same units as data; spread around mean |
| IQR | `Q3 − Q1` | robust spread, used for outlier fences |
| Percentile | value below which p % lie | robust position measure |

**Skewness:** mean > median → positively (right) skewed; mean < median → negatively (left) skewed; mean ≈ median → roughly symmetric.

**Empirical rule (normal distribution):** ≈ 68 % of values within μ ± 1σ, ≈ 95 % within ± 2σ, ≈ 99.7 % within ± 3σ.

---

## 4. Journal questions from the handout (with answers)

**Q1. What is statistics?**
The science of collecting, organising, summarising, presenting and analysing data, and drawing inferences from it. It is divided into **descriptive statistics** (summarising the observed data — mean, median, SD, plots) and **inferential statistics** (generalising from a sample to a population — hypothesis testing, confidence intervals, regression).

**Q2. Explain measures of central tendency with examples.**
A single value that represents the centre of the data.
- **Mean** — arithmetic average. Income [45k, 60k, 90k] → 65k. Best for symmetric numeric data.
- **Median** — middle value. [45k, 60k, 90k] → 60k. Best for skewed data/outliers.
- **Mode** — most frequent. Shoe size 8, 8, 9 → 8. Best for categorical data.

**Q3. Different types of variables with examples.**
- **Qualitative (categorical):** nominal (gender, colour, species — no order), ordinal (grades A/B/C, satisfaction low/medium/high — order, unequal gaps).
- **Quantitative (numeric):** discrete (number of students, count — whole numbers), continuous (height, income, temperature — any value in a range).
- Also: **ratio** scale (true zero, e.g. income) vs **interval** scale (no true zero, e.g. temperature in °C).

**Q4. Which method displays statistics of a DataFrame? Write the code.**
`df.describe()` displays count, mean, std, min, quartiles and max of all numeric columns. `df.describe(include="all")` also includes categorical columns (count, unique, top, freq). For grouped: `df.groupby("col")["num"].describe()`.

---

## 5. Viva questions the external examiner may ask

**1. Difference between mean, median and mode?**
Mean is the arithmetic average and uses every value. Median is the middle value after sorting and is robust to outliers. Mode is the most frequent value and is the only one applicable to nominal data.

**2. Which measure of central tendency is best for skewed data and why?**
Median, because it depends only on position, not magnitude, so extreme values do not pull it.

**3. When are mean and median equal?**
In a perfectly symmetric distribution (e.g. normal distribution).

**4. What does standard deviation tell you?**
The average distance of observations from the mean. SD = 0 means all values are identical; larger SD means more spread. It is in the same units as the data, unlike variance.

**5. Why divide by n−1 in sample standard deviation?**
Bessel's correction — the sample mean is estimated from the same data, so the deviations are slightly too small. Dividing by `n−1` makes the estimator **unbiased** for the population variance.

**6. Difference between variance and standard deviation?**
Variance is the average squared deviation (units²). Standard deviation is its square root, returning to the original units and therefore easier to interpret.

**7. What is a percentile? Give an example.**
The value below which a given percentage of observations fall. The 90th percentile of marks is the score below which 90 % of students lie.

**8. What is the IQR and why is it useful?**
`Q3 − Q1`, the spread of the middle 50 % of data. It is robust to outliers and is the basis of boxplots and the 1.5 × IQR outlier rule.

**9. What is an outlier?**
An observation that lies far from the rest of the data — outside `Q1 − 1.5·IQR` or `Q3 + 1.5·IQR`, or with |z| > 3.

**10. Difference between `describe()` and `info()`?**
`describe()` gives numerical summary statistics; `info()` gives structure — column names, dtypes, non-null counts and memory usage.

**11. What does `count` in `describe()` indicate?**
The number of **non-null** values. If `count` < total rows, that column has missing values.

**12. How do you get only the mean and median per group?**
`df.groupby("col")["num"].agg(["mean", "median"])` or `.agg(mean="mean", median="median")`.

**13. What is a box plot? What are its five parts?**
A plot of the five-number summary: minimum (whisker), Q1, median, Q3, maximum (whisker). Points beyond the whiskers are plotted individually as outliers.

**14. What is skewness? How do you detect it from mean and median?**
Asymmetry of a distribution. Mean > median → right/positive skew; mean < median → left/negative skew; mean ≈ median → symmetric.

**15. What is kurtosis?**
A measure of tail heaviness / peakedness relative to a normal distribution. Normal → 0 (excess kurtosis); > 0 → heavy tails, more outliers; < 0 → light tails.

**16. Can the mode be used for continuous data?**
Rarely, because exact values seldom repeat. You first group continuous data into bins (discretisation) and then take the modal class.

**17. How many modes can a dataset have?**
One (unimodal), two (bimodal), or many (multimodal). pandas' `mode()` returns all of them.

**18. Difference between descriptive and inferential statistics?**
Descriptive summarises the observed sample (mean, SD, charts). Inferential uses the sample to draw conclusions about the population (t-test, confidence interval, regression with p-values).

**19. What is the difference between population and sample standard deviation formula?**
Population: divide by `N`. Sample: divide by `n−1`. pandas `.std()` and `np.std(ddof=1)` give the sample value.

**20. How would you compute statistics for a column with missing values?**
pandas skips `NaN` automatically in `mean()`, `median()`, `std()`, `describe()`. Use `skipna=False` if you want `NaN` when missing values exist.

**21. What is the difference between `groupby().mean()` and `pivot_table()`?**
They produce the same numbers; `pivot_table` reshapes the result into a matrix (rows = one category, columns = another) with aggregation, and handles multiple categorical variables more naturally.

---

## 6. Common mistakes

- Saying `std` uses `n` — pandas uses `n − 1`.
- Reporting the mode without checking whether it is unique (`mode()` returns a Series).
- Forgetting `numeric_only=True` when calling `.median()` on a DataFrame that contains text columns (in newer pandas it may raise).
- Computing statistics on `age` when the handout asks for statistics on `income` grouped by `age_group`.
- Writing `df.groupby("age_group").describe()` and being unable to explain which columns the numbers belong to.
