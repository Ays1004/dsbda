# Practical A8 — Explanation & Viva Questions

## 1. What the practical is asking

1. Load the built-in **titanic** dataset with Seaborn.
2. Explore it to **find patterns** (survival vs sex/class/age, fare distribution).
3. Draw a **histogram of `fare`**.

The examiner mainly checks that you can choose the right plot and — more importantly — **read a conclusion out of it**.

---

## 2. Dataset

| Item | Detail |
|---|---|
| Rows | 891 passengers |
| Columns | 15: `survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `class`, `who`, `deck`, `embark_town`, `alive`, `alone`, `adult_male` |
| Target | `survived` (0 = died, 1 = survived) |
| Missing values | `age` 177, `deck` 688, `embarked` 2 |

`sns.load_dataset("titanic")` downloads the CSV from the seaborn-data GitHub repository and caches it in `~/seaborn-data`, so the first run needs internet.

---

## 3. Line-by-line explanation

### `sns.load_dataset("titanic")`

Returns a pandas DataFrame. Seaborn ships several standard datasets (tips, iris, titanic, penguins, flights, diamonds). It is **not** part of scikit-learn; `sklearn.datasets` has different built-ins (iris, digits, wine, breast_cancer).

### Pattern-finding with `groupby`

```python
df.groupby("sex")["survived"].mean()
```

`survived` is 0/1, so its **mean = survival rate**. Grouping by `sex` gives 0.742 for females and 0.188 for males. This numeric summary is what you quote in "observations".

```python
df.groupby(["sex", "class"])["survived"].mean()
```

Two grouping keys produce a MultiIndex — survival rate for each sex–class combination (e.g. 3rd-class male ≈ 0.135, 1st-class female ≈ 0.968).

### `sns.histplot`

```python
sns.histplot(x="fare", bins=10, data=df)
```

- Bins partition the range of `fare` into 10 equal-width intervals and count the values in each.
- `bins` controls resolution: too few hides shape, too many shows noise.
- `kde=True` overlays a smooth kernel density estimate.
- `hue="survived"` colours by a second variable; `multiple="stack"` stacks the bars, `"dodge"` places them side by side.

**Reading the fare histogram:** the bars are tallest at low fares and the tail stretches to 512. This is a **right-skewed (positively skewed)** distribution. Most passengers paid under 50; a few paid very high first-class fares. For such data the **median** is a better summary than the mean, and a **log transform** would make the distribution more symmetric.

### `sns.boxplot(x="class", y="age", data=df)`

For each class, the box shows Q1–Q3, the line inside is the median, the whiskers extend to 1.5 × IQR, and dots are outliers. This answers "how does age vary across passenger classes?" in one picture.

### `sns.violinplot`

A box plot plus a mirrored KDE — it shows **where the data is dense**, not just the quartiles. A bimodal violin reveals two subgroups that a box plot hides.

### `sns.countplot(x="sex", hue="survived", data=df)`

Counts rows per category. Useful for class frequencies (how many male vs female, how many survived in each group).

### `sns.scatterplot(x="age", y="fare", hue="survived")`

Shows the relationship between two numeric variables, with a third variable as colour. Here the relationship is weak; the useful information is that survivors (orange) cluster at higher fares.

### `sns.heatmap(df.corr(numeric_only=True), annot=True)`

The correlation matrix shows pairwise linear correlation coefficients in [−1, 1]. Strong positive: `fare`–`pclass` (higher class number = cheaper, so actually `pclass` is inverse-coded), `sibsp`–`parch` (families travel together). `survived` correlates positively with `fare` and negatively with `pclass` (numerically).

### `sns.pairplot(..., hue="survived")`

Draws scatter plots for every pair of numeric columns plus histograms on the diagonal, coloured by survival. It is a quick multivariate EDA; on 891 rows it is slow but acceptable.

---

## 4. Journal questions from the handout (with answers)

**Q1. List the different types of plots used to find patterns in data.**
- **Distribution:** histogram, KDE/distplot, box plot, violin plot.
- **Categorical:** count plot, bar plot, point plot, swarm plot, strip plot.
- **Relationship:** scatter plot, line plot, pair plot, heat map (correlation), joint plot, regression plot (`regplot`/`lmplot`).
- **Composition:** pie chart, stacked bar, area chart.
- **Multivariate:** pair plot, heat map, `FacetGrid`/`catplot` with `hue` and `col`.

**Q2. When do you use distribution plots and when categorical plots?**
Distribution plots (histogram, KDE, box, violin) answer "how are the values of a **numeric** variable spread?" — shape, centre, spread, skew, outliers. Categorical plots (count, bar, swarm, strip, point) answer "how does a **categorical** variable relate to a numeric variable or to counts?" — comparisons between groups. Rule: numeric variable alone/against a category → distribution plot; category frequency or category vs numeric summary → categorical plot.

**Q3. Conclusion from the swarm plot (titanic).**
A swarm plot shows every observation, so the conclusion should mention both the group comparison and the density: e.g. "female passengers had a much higher survival rate than males, and the female survivors are concentrated in the younger age range; 3rd-class passengers form the largest and most spread-out cluster." In general: quote the direction of the difference, the group with the highest/lowest value, and any visible overlap or outliers.

**Q4. Which parameter adds another categorical variable to a violin plot?**
`hue`. Example:
```python
sns.violinplot(x="class", y="age", hue="sex", data=df, split=True)
```
`hue="sex"` draws a separate violin for each sex within each class, and `split=True` merges them into one half-violin per category for direct comparison.

---

## 5. Viva questions the external examiner may ask

**1. What is data visualisation? Why is it needed?**
The graphical representation of data to reveal patterns, trends, outliers and relationships that are hard to see in tables. It supports EDA, communicates results, and guides model selection.

**2. Difference between matplotlib and seaborn?**
Matplotlib is the low-level plotting library (full control, more code). Seaborn is built on matplotlib and provides high-level statistical plots with DataFrame integration, automatic legends, `hue` grouping and attractive default themes.

**3. What does `hue` do?**
Maps a categorical variable to colour, splitting the plot into groups (e.g. survived vs not) within the same axes.

**4. Difference between `displot` and `histplot`?**
`histplot` is an **axes-level** function (draws on one matplotlib Axes, can be placed in subplots). `displot` is a **figure-level** function (manages its own figure, supports `col`/`row` faceting, returns a FacetGrid). `distplot` is the deprecated older function.

**5. What does `bins` control in a histogram?**
The number of intervals the data range is divided into. More bins = finer detail but noisier; fewer bins = smoother but may hide shape. Common rules: Sturges, Freedman–Diaconis, or the square-root rule.

**6. What is a KDE?**
Kernel Density Estimate — a smoothed, non-parametric estimate of the probability density function, obtained by placing a kernel (usually Gaussian) on each observation and summing.

**7. How is a box plot drawn? What are its parts?**
Box from Q1 to Q3 (IQR), line at the median, whiskers to the most extreme values within 1.5 × IQR, and individual points beyond the whiskers as outliers. It displays the five-number summary.

**8. Box plot vs violin plot?**
Box plot shows only the five-number summary; violin also shows the full density shape (can be bimodal). Violin is better for distribution comparison, box is more compact and easier to read.

**9. What is a count plot?**
A bar chart of the frequency of each category (`sns.countplot`). It is a univariate categorical plot.

**10. What does a scatter plot show?**
The relationship between two numeric variables; patterns, clusters, correlation and outliers.

**11. What is a pair plot?**
A grid of scatter plots for every pair of numeric variables with histograms/KDEs on the diagonal; used for multivariate EDA.

**12. Why did the fare distribution look skewed?**
Because a small number of first-class passengers paid very high fares while most passengers paid low fares, producing a long right tail. Mean > median confirms positive skew.

**13. How do you handle a skewed variable before modelling?**
Log/sqrt transform, or use robust models and report median/IQR.

**14. What does `df.corr()` compute?**
The Pearson correlation coefficient between numeric columns — linear association in [−1, 1]. Use `method="spearman"` for monotonic non-linear association.

**15. What conclusion can you draw about survival?**
Sex is the strongest factor (female survival ≈ 74 % vs male ≈ 19 %), followed by class (1st ≈ 63 %, 3rd ≈ 24 %), consistent with the "women and children first" evacuation policy. Age alone is a weaker discriminator.

**16. Is correlation causation?**
No. Correlation only measures association; a third variable (confounding) or reverse causation may explain it.

**17. What is EDA?**
Exploratory Data Analysis — the process of summarising and visualising data (univariate, bivariate, multivariate) to understand distributions, relationships and anomalies before modelling.

**18. How do you save a seaborn figure to a file?**
`plt.savefig("plot.png", dpi=300, bbox_inches="tight")` before `plt.show()`. For figure-level functions: `g.savefig("plot.png")`.

**19. Why is `plt.show()` needed?**
It renders the figure in interactive environments. In scripts, figures are not displayed without it (and with the Agg backend they are not displayed at all).

**20. What is the difference between `pclass` and `class` in the titanic dataset?**
`pclass` is the numeric 1/2/3 class; `class` is the same information as an ordered categorical label ("First", "Second", "Third"). They are redundant — do not use both as features.

---

## 6. Common mistakes

- Forgetting `dropna()` for `age` in `histplot`; seaborn usually handles NaN silently, but manual calculations may fail.
- Calling `sns.distplot` in a new seaborn version — it is deprecated; use `histplot`/`displot`.
- Saying the mean is the best summary of `fare` despite the obvious skew.
- Plotting `pclass` and `class` as if they were independent features.
- Not writing observations — the handout explicitly asks for patterns, not just plots.
- Using a pie chart for 891 values / many categories.
