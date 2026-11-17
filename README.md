# DSBDA — Data Science and Big Data Analytics Practicals

SPPU TE Computer Engineering, Semester 6 — lab practical notes.

Each practical folder contains exactly **two files**:

| File | Contents |
|---|---|
| `CODE.md` | Complete runnable code + expected output + journal notes |
| `EXPLANATION.md` | Line-by-line explanation + handout questions + viva questions with answers |

---

## Group A — Data Science (Python)

| # | Practical | Folder |
|---|---|---|
| A1 | Data Wrangling-I | [Practical-A1-Data-Wrangling-I](Practical-A1-Data-Wrangling-I) |
| A2 | Data Wrangling-II | [Practical-A2-Data-Wrangling-II](Practical-A2-Data-Wrangling-II) |
| A3 | Descriptive Statistics | [Practical-A3-Descriptive-Statistics](Practical-A3-Descriptive-Statistics) |
| A4 | Data Analytics-I — Linear Regression | [Practical-A4-Linear-Regression](Practical-A4-Linear-Regression) |
| A5 | Data Analytics-II — Logistic Regression | [Practical-A5-Logistic-Regression](Practical-A5-Logistic-Regression) |
| A6 | Data Analytics-III — Naïve Bayes | [Practical-A6-Naive-Bayes](Practical-A6-Naive-Bayes) |
| A7 | Text Analytics | [Practical-A7-Text-Analytics](Practical-A7-Text-Analytics) |
| A8 | Data Visualization-I — Titanic | [Practical-A8-Data-Visualization-I](Practical-A8-Data-Visualization-I) |
| A9 | Data Visualization-II — Box plot | [Practical-A9-Data-Visualization-II](Practical-A9-Data-Visualization-II) |
| A10 | Data Visualization-III — Iris | [Practical-A10-Data-Visualization-III](Practical-A10-Data-Visualization-III) |

## Group B — Big Data (Hadoop / Spark)

| # | Practical | Folder |
|---|---|---|
| B1 | Hadoop WordCount (MapReduce) | [Practical-B1-Hadoop-WordCount](Practical-B1-Hadoop-WordCount) |
| B2 | MapReduce Log Analysis | [Practical-B2-MapReduce-Log-Analysis](Practical-B2-MapReduce-Log-Analysis) |
| B4 | Scala with Apache Spark | [Practical-B4-Spark-Scala](Practical-B4-Spark-Scala) |

---

## Datasets used

The code expects the CSV file in the same folder as the script/notebook.

| Practical | File | Source |
|---|---|---|
| A1, A3, A6, A10 | `iris.csv` | <https://www.kaggle.com/datasets/uciml/iris> |
| A4 | `Assignment-A4-BostonHousing.csv` | <https://www.kaggle.com/c/boston-housing> |
| A5 | `Assignment-A5-Social_Network_Ads.csv` | <https://www.kaggle.com/datasets/akram24/social-network-ads> |
| A2 | generated in code | — |
| A7 | text written in code | — |
| A8, A9 | `sns.load_dataset("titanic")` | downloaded by seaborn (needs internet once) |
| B1, B2 | `input.txt`, `logfile.txt` | provided in the respective `CODE.md` |

---

## Quick setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

For A7, the NLTK downloads run inside the script (`nltk.download(...)`).

For B1/B2 you need Hadoop installed and running (`start-all.sh`), and Java (`javac`, `hadoop classpath`).
For B4 you need Scala and Spark (`spark-shell`).

---

## Note

These notes cover the SPPU DSBDA lab assignments (A1–A10, B1, B2, B4). B3/B5+ are not part of this set. Always cross-check the problem statement in your official handout before the practical.
