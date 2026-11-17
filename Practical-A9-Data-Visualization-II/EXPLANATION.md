# Practical A9 — Explanation & Viva Questions

## 1. What the practical is asking

Draw **one specific box plot**: `age` on the y-axis, `sex` on the x-axis, and `survived` as a second grouping (colour), then **write observations**. The marks are mostly in the observations — the code is four lines.

---

## 2. The one line that matters

```python
sns.boxplot(x="sex", y="age", hue="survived", data=df, palette="viridis")
```

- `x="sex"` → one box group per gender on the x-axis.
- `y="age"` → the numeric variable whose distribution is drawn.
- `hue="survived"` → within each gender, a separate box for survived (1) and died (0).
- `data=df` → the DataFrame (allows using column names as strings).

The plot therefore has **four boxes**: female/died, female/survived, male/died, male/survived.

---

## 3. How a box plot is constructed

For each group:

1. Sort the values.
2. `Q1` = 25th percentile, `Q2` = median (50th), `Q3` = 75th percentile.
3. `IQR = Q3 − Q1`.
4. Whiskers extend to the smallest/largest value still inside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
5. Values outside the whiskers are drawn as individual **outlier** dots.

So one box encodes five numbers plus the outliers — that is why it is called the **five-number summary** plot.

---

## 4. The mean/median/mode block

The code computes them by hand to satisfy the handout's statistics requirement:

```python
sorted_age = sorted(age_data)
n = len(sorted_age)
median_age = sorted_age[n // 2] if n % 2 else (sorted_age[n//2 - 1] + sorted_age[n//2]) / 2
mode_age = Counter(age_data).most_common(1)[0][0]
```

Result: **mean 29.70, median 28, mode 24** (for the 714 non-null ages).
`mean > median` → slight **positive (right) skew**, which is also visible in the age histogram.

`Counter(...).most_common(1)` returns the most frequent `(value, count)` pair. pandas equivalents are `df["age"].mean()`, `.median()`, `.mode()`.

---

## 5. The grouped statistics table (quote these in the viva)

```
sex     survived   count   mean    median
female  0            64    25.05    24.5
female  1           197    28.85    28.0
male    0           360    31.62    29.0
male    1            93    27.28    28.0
```

Survival rates: **female 74.2 %**, **male 18.9 %**.

**Observations:**
- Sex is the strongest visible factor; the age distributions of survivors and non-survivors overlap, so age alone does not explain survival.
- Female survivors are slightly *older* on average than female non-survivors (28.9 vs 25.0) — because most women survived, and the deaths were concentrated among younger women in 3rd class.
- Male survivors are slightly *younger* than male non-survivors (27.3 vs 31.6), consistent with children/young men being prioritized.
- All groups have outliers at the upper end (elderly passengers), and the male groups extend furthest (74–80 years).
- 177 ages are missing, which is a limitation of this analysis.

---

## 6. Journal questions from the handout (with answers)

**Q1. Explain Exploratory Data Analysis (EDA).**
EDA is the process of summarising and visualising a dataset before modelling to understand its main characteristics: distributions, central tendency, spread, missing values, outliers and relationships between variables. It has three levels — **univariate** (one variable: histogram, box plot), **bivariate** (two variables: scatter, grouped box plot, correlation) and **multivariate** (more than two: pair plot, heat map, `hue` grouping). EDA guides feature selection, detects data-quality problems, and suggests which models are appropriate.

**Q2. Explain univariate analysis.**
Analysis of a single variable at a time. For numeric variables: mean, median, mode, variance, SD, skew, histogram, box plot. For categorical variables: frequency table, count plot, bar chart, mode. Purpose: understand the variable's distribution, spot outliers and missing values, decide on transformations.

**Q3. What is multivariate analysis? Explain.**
Analysis of three or more variables simultaneously to find relationships and interactions. Examples: a box plot of `age` by `sex` with `hue="survived"` (two categorical + one numeric); a correlation heat map; a pair plot; multiple regression with several predictors. Purpose: detect confounding, interactions and combined effects that univariate/bivariate analysis cannot show.

**Q4. How are distplot and boxplot created?**
```python
# Distribution plot (histogram + KDE)
sns.histplot(df["age"].dropna(), kde=True)     # older seaborn: sns.distplot(df["age"].dropna())

# Box plot
sns.boxplot(x="sex", y="age", hue="survived", data=df)
plt.show()
```
`histplot` bins the numeric values and (with `kde=True`) overlays a smoothed density. `boxplot` groups by the categorical variable(s) and draws the five-number summary per group, with points beyond the whiskers as outliers.

---

## 7. Viva questions the external examiner may ask

**1. What is a box plot?**
A plot of the five-number summary — minimum (whisker), Q1, median, Q3, maximum (whisker) — with points beyond 1.5 × IQR shown individually as outliers. It compares distributions across categories compactly.

**2. What does the box represent? What does the line inside represent?**
The box spans Q1 to Q3 (the interquartile range, middle 50 % of data). The line inside is the median (Q2).

**3. What do the whiskers represent?**
The most extreme values that are still within `1.5 × IQR` of the box. They are **not** the min/max unless no outliers exist.

**4. How are outliers identified in a box plot?**
Values below `Q1 − 1.5·IQR` or above `Q3 + 1.5·IQR` are plotted as individual dots. Extreme outliers at 3 × IQR are sometimes shown separately.

**5. Why is a box plot preferred over a histogram when comparing groups?**
One histogram per group would need several subplots and is hard to compare visually. A box plot puts all groups on the same axes and highlights median, spread and outliers at a glance.

**6. What is the limitation of a box plot?**
It hides the shape of the distribution (bimodality, multi-modality) and the exact sample size; two very different distributions can produce identical boxes. Violin or swarm plots reveal the shape.

**7. What does `hue` do in this plot?**
It adds a second categorical dimension (`survived`) inside each x-category, producing separate coloured boxes per gender.

**8. What is the difference between `boxplot` and `violinplot`?**
The violin adds a KDE showing where values are dense; the box only shows quartiles. `violinplot(..., split=True)` with two hue levels draws half-violins for direct comparison.

**9. What is `catplot` used for?**
It is the figure-level interface for categorical plots, with `kind="count"`, `"bar"`, `"box"`, `"violin"`, `"strip"`, `"swarm"`, `"point"`. It supports faceting (`row`, `col`) and returns a FacetGrid.

**10. Difference between `countplot` and `barplot`?**
`countplot` plots the **number of observations** per category. `barplot` plots an **aggregate (default mean) of a numeric variable** per category, with a confidence interval.

**11. What is the difference between a histogram and a bar chart?**
Histogram = continuous numeric data binned into intervals, bars touch. Bar chart = categorical data, bars are separate.

**12. How is the median found for an even number of observations?**
Average of the two middle values after sorting. The code implements this: `(sorted[n//2 - 1] + sorted[n//2]) / 2`.

**13. What are the mean, median and mode of `age` here?**
Mean 29.70, median 28, mode 24. Mean > median indicates slight right skew.

**14. What conclusions can you draw from the required box plot?**
Sex is the strongest factor in survival (female 74 % vs male 19 %); age overlaps heavily between survivors and non-survivors, so age alone is a weak predictor, although children appear more frequently in the survivor group. The plot also exposes elderly outliers and missing-data limitations.

**15. Why is `age` missing for so many rows, and how does seaborn handle it?**
177 values are missing in the dataset. Seaborn's statistical functions drop missing values for the plotted variable automatically (like pandas `dropna()`), so each box uses only the available records.

**16. What is the difference between `plt.show()` and `plt.savefig()`?**
`show()` displays the figure; `savefig()` writes it to a file. Call `savefig` before `show`, because `show` can clear the current figure.

**17. Is the difference between male and female survival statistically significant?**
Visually yes, but to claim significance you need a test — e.g. a chi-square test of independence between `sex` and `survived` (or a two-proportion z-test). The plot alone only shows association.

**18. What is a confounding variable? Give an example here.**
A third variable that influences both the apparent cause and the effect. `pclass` confounds sex and survival: more females travelled 1st/2nd class, and those classes had better access to lifeboats. So part of the sex effect may actually be a class effect.

**19. What is the IQR?**
`Q3 − Q1`, the range of the middle 50 % of the data. It is robust to outliers and is the basis of the box-plot fences.

**20. Why might a violin plot show a bimodal shape for age?**
Because the data contains two subgroups — e.g. children travelling with families and young adults — so the density has two peaks. A box plot would not reveal this.

---

## 8. Common mistakes

- Plotting `age` vs `survived` only, without `sex` — the handout explicitly requires both.
- Writing "the box shows min to max" — the box is Q1 to Q3, whiskers are 1.5 × IQR.
- Forgetting to mention missing values or the overlapping distributions in the observations.
- Treating the box-plot dots as "errors" — they are legitimate high-age observations, not data errors.
- Using a bar chart/mean-only summary and claiming it shows the distribution.
