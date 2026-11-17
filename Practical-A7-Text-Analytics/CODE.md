# Practical A7 — Text Analytics

**Aim:** Perform document preprocessing — tokenisation, POS tagging, stop-word removal, stemming and lemmatisation — and represent documents using Term Frequency–Inverse Document Frequency (TF-IDF).

**Dataset:** Sample text written inside the program (no external file required).
**Libraries:** `nltk`, `scikit-learn`, `re`

---

## Full Code

```python
# ============================================================
# Practical A7 : Text Analytics
# Part 1 - Preprocessing | Part 2 - TF-IDF
# ============================================================

import re
import math
import nltk
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------------- 0. Download the required NLTK resources (one time) ----------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("tagsets")

# ---------------- 1. Sample document ----------------
text = (
    "Hello everyone! I am learning Data Science and Big Data Analytics. "
    "The students are studying hard because they want to score well. "
    "Python is easier than Java, and it is widely used for analytics."
)
print("Original text:\n", text)

# ---------------- 2. Sentence tokenisation ----------------
sentences = sent_tokenize(text)
print("\nSentence tokens:")
for i, s in enumerate(sentences, 1):
    print(f"  {i}. {s}")

# ---------------- 3. Word tokenisation ----------------
words = word_tokenize(text)
print("\nWord tokens:\n", words)
print("Number of word tokens:", len(words))

# ---------------- 4. Remove punctuation and special characters ----------------
clean_text = re.sub(r"[^a-zA-Z]", " ", text)   # keep only letters, replace rest by space
print("\nAfter removing punctuation:\n", clean_text)

# ---------------- 5. Lowercase + remove stop words ----------------
stop_words = set(stopwords.words("english"))
print("\nNumber of English stop words:", len(stop_words))

tokens = word_tokenize(clean_text.lower())
filtered_words = [w for w in tokens if w not in stop_words]
print("\nAfter stop-word removal:\n", filtered_words)

# ---------------- 6. Stemming (Porter stemmer) ----------------
ps = PorterStemmer()
stem_words = ["write", "writing", "wrote", "writes", "reading", "reads",
              "studies", "studying", "cries", "cry", "easily", "fairly"]
print("\nStemming:")
for w in stem_words:
    print(f"  {w:10s} -> {ps.stem(w)}")

# ---------------- 7. Lemmatisation (WordNet) ----------------
lemmatizer = WordNetLemmatizer()
text2 = "studies studying cries cry"
tokens2 = word_tokenize(text2)
print("\nLemmatisation (default, noun):")
for w in tokens2:
    print(f"  {w:10s} -> {lemmatizer.lemmatize(w)}")

print("\nLemmatisation (verb-based, pos='v'):")
for w in tokens2:
    print(f"  {w:10s} -> {lemmatizer.lemmatize(w, pos='v')}")

# ---------------- 8. POS tagging ----------------
pos_text = "Hello everyone this is a sample text. Earth is the third planet."
pos_tokens = word_tokenize(pos_text)
pos_tags = nltk.pos_tag(pos_tokens)
print("\nPOS tagging:")
for word, tag in pos_tags:
    print(f"  {word:10s} -> {tag}")

# ============================================================
# PART 2 : TF-IDF
# ============================================================

# ---------------- 9. Documents from the handout question ----------------
docA = "Jupiter is the largest Planet"
docB = "Mars is the fourth planet from the Sun"

# lowercase + remove stop words + tokenise
def preprocess(doc):
    doc = doc.lower()
    toks = word_tokenize(doc)
    return [t for t in toks if t.isalpha() and t not in stop_words]

tokensA = preprocess(docA)
tokensB = preprocess(docB)
print("\nTokens A:", tokensA)
print("Tokens B:", tokensB)

# ---------------- 10. Manual TF, IDF, TF-IDF (handout algorithm) ----------------
documents = {"A": tokensA, "B": tokensB}
vocabulary = sorted(set(tokensA) | set(tokensB))
N = len(documents)

def term_frequency(tokens, word):
    """TF = count of the word / total words in that document"""
    return tokens.count(word) / len(tokens)

def inverse_document_frequency(word):
    """IDF = log(N / df), df = number of documents containing the word"""
    df = sum(1 for toks in documents.values() if word in toks)
    return 0.0 if df == 0 else math.log(N / df)

rows = []
for word in vocabulary:
    tf_a = term_frequency(tokensA, word)
    tf_b = term_frequency(tokensB, word)
    idf = inverse_document_frequency(word)
    rows.append({"word": word, "TF_A": round(tf_a, 3), "TF_B": round(tf_b, 3),
                 "IDF": round(idf, 3),
                 "TFIDF_A": round(tf_a * idf, 3),
                 "TFIDF_B": round(tf_b * idf, 3)})

tfidf_table = pd.DataFrame(rows)
print("\nManual TF-IDF table:\n", tfidf_table.to_string(index=False))

# ---------------- 11. TF-IDF with sklearn ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [docA, docB]
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfidf_matrix = vectorizer.fit_transform(corpus)

print("\nFeature names (vocabulary):", vectorizer.get_feature_names_out())
print("\nTF-IDF matrix (documents x words):\n", tfidf_matrix.toarray().round(3))

print("\nAs a readable table:")
print(pd.DataFrame(tfidf_matrix.toarray().round(3),
                   index=["Document A", "Document B"],
                   columns=vectorizer.get_feature_names_out()))
```

## Output (abridged)

```text
Sentence tokens:
  1. Hello everyone!
  2. I am learning Data Science and Big Data Analytics.
  3. The students are studying hard because they want to score well.
  4. Python is easier than Java, and it is widely used for analytics.

Word tokens:
['Hello', 'everyone', '!', 'I', 'am', 'learning', 'Data', 'Science', 'and',
 'Big', 'Data', 'Analytics', '.', 'The', 'students', 'are', 'studying', ...]
Number of word tokens: 39

After stop-word removal:
['hello', 'everyone', 'learning', 'data', 'science', 'big', 'data',
 'analytics', 'students', 'studying', 'hard', 'want', 'score', 'well',
 'python', 'easier', 'java', 'widely', 'used', 'analytics']

Stemming:
  write      -> write
  writing    -> write
  wrote      -> wrote
  writes     -> write
  reading    -> read
  studies    -> studi
  studying   -> studi
  cries      -> cri
  cry        -> cri
  easily     -> easili

Lemmatisation (default, noun):     Lemmatisation (verb-based):
  studies    -> study                studies    -> study
  studying   -> studying             studying   -> study
  cries      -> cry                  cries      -> cry
  cry        -> cry                  cry        -> cry

POS tagging:
  Hello      -> NNP      everyone   -> NN
  this       -> DT       is         -> VBZ
  a          -> DT       sample     -> JJ
  text       -> NN       Earth      -> NN

Manual TF-IDF table:
    word  TF_A  TF_B   IDF  TFIDF_A  TFIDF_B
 fourth 0.000  0.25 0.693    0.000    0.173
jupiter 0.333  0.00 0.693    0.231    0.000
largest 0.333  0.00 0.693    0.231    0.000
   mars 0.000  0.25 0.693    0.000    0.173
 planet 0.333  0.25 0.000    0.000    0.000
    sun 0.000  0.25 0.693    0.000    0.173

sklearn TF-IDF matrix:
            fourth  jupiter  largest   mars  planet    sun
Document A   0.000    0.632    0.632  0.000   0.449  0.000
Document B   0.534    0.000    0.000  0.534   0.380  0.534
```

The difference between the manual and sklearn values is expected — sklearn uses smoothed IDF `ln((1+N)/(1+df)) + 1` and L2-normalises each row. The conclusion is identical: `planet` gets zero weight because it appears in both documents.

---

## Handout question answer (write it in the journal)

For **A = "Jupiter is the largest Planet"** and **B = "Mars is the fourth planet from the Sun"**, after lowercasing and removing stop words:

| Word | TF in A | TF in B | df | IDF = log(2/df) | TF-IDF A | TF-IDF B |
|---|---|---|---|---|---|---|
| jupiter | 1/3 = 0.333 | 0 | 1 | 0.693 | 0.231 | 0 |
| largest | 1/3 = 0.333 | 0 | 1 | 0.693 | 0.231 | 0 |
| planet | 1/3 = 0.333 | 1/4 = 0.250 | 2 | 0 | 0 | 0 |
| mars | 0 | 1/4 = 0.250 | 1 | 0.693 | 0 | 0.173 |
| fourth | 0 | 1/4 = 0.250 | 1 | 0.693 | 0 | 0.173 |
| sun | 0 | 1/4 = 0.250 | 1 | 0.693 | 0 | 0.173 |

`planet` appears in **both** documents, so its IDF is zero — a word common to the corpus carries no discriminating information. That is the whole idea of TF-IDF in one sentence.
