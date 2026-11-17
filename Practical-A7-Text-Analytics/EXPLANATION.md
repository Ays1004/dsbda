# Practical A7 — Explanation & Viva Questions

## 1. What the practical is asking

Two parts:

1. **Preprocessing** a sample document: tokenisation (sentence + word), POS tagging, stop-word removal, stemming and lemmatisation.
2. **Representing** the document numerically with **TF-IDF** (Term Frequency × Inverse Document Frequency).

The handout also gives the manual TF-IDF algorithm steps and a specific question on documents A and B — that hand calculation is very likely to be asked.

---

## 2. Line-by-line explanation

### NLTK resources

```python
nltk.download("punkt")          # sentence/word tokenizer models
nltk.download("stopwords")      # list of English stop words
nltk.download("wordnet")        # lexical database for lemmatisation
nltk.download("averaged_perceptron_tagger")   # POS tagger
```

NLTK is modular — the algorithms ship with the library but the trained models/corpora are downloaded separately. `punkt_tab` is required by newer NLTK versions, and `averaged_perceptron_tagger_eng` is the current English tagger model; older versions only have `averaged_perceptron_tagger`.

### Tokenisation

```python
sent_tokenize(text)   # splits into sentences, handles "Dr.", "U.S.A." etc.
word_tokenize(text)   # splits into words AND punctuation
```

**Token** = the smallest meaningful unit of text. Tokenisation is the first step of every NLP pipeline because every later step works on tokens, not raw strings.

### Punctuation removal

```python
re.sub(r"[^a-zA-Z]", " ", text)
```

`[^a-zA-Z]` means "any character that is **not** a letter"; `sub` replaces it with a space. This keeps only alphabetic tokens and removes digits, punctuation and symbols. (`re.sub(r"\W+", " ", text)` is the equivalent "non-word character" form.)

### Stop-word removal

```python
stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w not in stop_words]
```

Stop words are extremely frequent words with little semantic content: *is, the, a, an, and, of, to, in, that, it*. Removing them reduces the vocabulary and lets the model focus on meaningful terms. **Caveat:** in sentiment analysis, removing "not" destroys negation — say this if asked about limitations.

### Stemming

```python
ps = PorterStemmer()
ps.stem("studies")   # -> studi
```

Stemming chops affixes using fixed rules. The result **may not be a real word** ("studi", "cri"). Algorithms: Porter (most common), Lancaster (more aggressive), Snowball (improved Porter).

### Lemmatisation

```python
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("studies")            # -> study   (default pos='n')
lemmatizer.lemmatize("studying", pos="v")  # -> study
```

Lemmatisation uses a dictionary (WordNet) and the POS of the word, so it returns a **valid dictionary word** (the lemma). It is slower but more accurate.

### Stemming vs Lemmatisation — the exact table for the handout question

Text: `"studies studying cries cry"`

| Word | Porter stem | Lemma (noun) | Lemma (verb) |
|---|---|---|---|
| studies | studi | study | study |
| studying | studi | studying | study |
| cries | cri | cry | cry |
| cry | cri | cry | cry |

**Comment:** both `studies` and `studying` reduce to the same stem `studi`, which is not an English word — stemming ignores meaning and context. Lemmatisation with `pos='v'` gives the correct base form `study` for both. Stemming merged `cries` and `cry` into `cri` too, but lemmatisation produces the real word `cry`. So stemming is faster and cruder; lemmatisation is slower but linguistically correct. The difference is visible: **stem "studi" is not a word, lemma "study" is.**

### POS tagging

```python
nltk.pos_tag(word_tokenize(text))
```

Each token gets a Penn Treebank tag:

| Tag | Meaning | Example |
|---|---|---|
| NN | noun, singular | planet |
| NNS | noun, plural | students |
| NNP | proper noun, singular | Earth |
| VB | verb, base form | study |
| VBD | verb, past tense | studied |
| VBG | verb, present participle | studying |
| JJ | adjective | largest |
| RB | adverb | easily |
| PRP | personal pronoun | I, he |
| DT | determiner | the, a |
| IN | preposition | from, in |
| CC | coordinating conjunction | and |
| CD | cardinal number | three |

POS tags are needed for lemmatisation (you must know whether "studying" is a verb or a noun), named-entity recognition and parsing.

---

## 3. TF-IDF theory

### Definitions

- **Document (d)** — one text unit.
- **Corpus (D)** — the collection of documents, `N = |D|`.
- **Term (t)** — a word/token.
- **df(t)** — number of documents containing `t`.
- **TF(t, d)** — how often `t` appears in `d`: `count(t, d) / total terms in d`.
- **IDF(t)** — `log(N / df(t))`.
- **TF-IDF** — `TF(t, d) × IDF(t)`.

### Why IDF?

A word that appears in every document (`planet`) cannot distinguish documents, so its IDF is `log(2/2) = 0`. A word that appears in only one document (`jupiter`) gets a high weight `log(2/1) = 0.693`. IDF therefore **down-weights common words and boosts rare, discriminative words**.

### Manual calculation (handout question)

Documents after lowercasing and stop-word removal:

```
A: jupiter, largest, planet        (3 terms)
B: mars, fourth, planet, sun       (4 terms)
N = 2
```

| Word | TF_A | TF_B | df | IDF | TF-IDF A | TF-IDF B |
|---|---|---|---|---|---|---|
| jupiter | 1/3 = 0.333 | 0 | 1 | log(2/1) = 0.693 | 0.231 | 0 |
| largest | 1/3 = 0.333 | 0 | 1 | 0.693 | 0.231 | 0 |
| planet | 1/3 = 0.333 | 1/4 = 0.25 | 2 | log(2/2) = 0 | 0 | 0 |
| mars | 0 | 0.25 | 1 | 0.693 | 0 | 0.173 |
| fourth | 0 | 0.25 | 1 | 0.693 | 0 | 0.173 |
| sun | 0 | 0.25 | 1 | 0.693 | 0 | 0.173 |

### Why sklearn's numbers are different

```python
TfidfVectorizer(stop_words="english")
```

sklearn uses **smoothed IDF** and **L2 normalisation**:

```
idf(t) = ln((1 + N) / (1 + df(t))) + 1
tf-idf = tf × idf, then each document vector is divided by its L2 norm
```

With `N = 2`: `idf(jupiter) = ln(3/2) + 1 = 1.405`, `idf(planet) = ln(3/3) + 1 = 1`.
For document A the vector `[1.405, 1.405, 1]` has norm `2.224`, so the values become `0.632, 0.632, 0.449`.

Both are "TF-IDF"; the manual version is the textbook formula, sklearn's is the engineered, numerically stable version. If the examiner asks why they differ, this is the answer.

---

## 4. Journal questions from the handout (with answers)

**Q1. Explain the basic concepts of text analytics.**
Text analytics (text mining) extracts structured, actionable information from unstructured text. Pipeline: (i) **acquisition** — collect documents; (ii) **preprocessing** — tokenisation, lowercasing, punctuation/stop-word removal, stemming/lemmatisation; (iii) **representation** — Bag of Words, TF-IDF, embeddings; (iv) **analysis** — classification, clustering, sentiment analysis, topic modelling (LDA), NER; (v) **interpretation/visualisation** — word clouds, dashboards. Applications: spam filtering, sentiment analysis, resume screening, chatbots, fake-news detection.

**Q2. Explain Inverse Document Frequency in detail.**
IDF measures how much information a term carries. `IDF(t) = log(N / df(t))`. If a term occurs in many documents, `df` is large, so IDF is small — the term is common and uninformative ("the", "is"). If it occurs in few documents, IDF is large — the term is discriminative ("jupiter"). Multiplying by TF gives a weight that is high only when a term is both frequent **in this document** and rare **across the corpus**. A term present in every document gets `log(1) = 0`.

**Q3. Stemming vs lemmatisation for "studies studying cries cry"** → see the table above. Stemming gives *studi, studi, cri, cri* (crude, non-words, and it cannot distinguish noun/verb). Lemmatisation with `pos='v'` gives *study, study, cry, cry* (valid words, context-aware). Conclusion: stemming is faster but over-merges; lemmatisation is slower but semantically correct.

**Q4. Code for stop-word removal + TF/IDF/TF-IDF for A and B** → the code is in `CODE.md`, and the numeric answer is the table above.

---

## 5. Viva questions the external examiner may ask

**1. What is NLP?**
Natural Language Processing — the field that enables computers to understand, interpret and generate human language. Text analytics is the applied, data-extraction side of NLP.

**2. What is tokenisation? Name its types.**
Splitting text into smaller units. Types: sentence tokenisation, word tokenisation, character tokenisation, subword tokenisation (BPE, WordPiece — used by transformers).

**3. Why do we convert text to lowercase?**
So that "Planet", "planet" and "PLANET" are treated as the same term; otherwise the vocabulary inflates and counts split across variants.

**4. What are stop words? Why remove them?**
Very frequent grammatical words with little meaning. Removing them shrinks the feature space and reduces noise. Downside: negation ("not good") and phrase meaning can be lost.

**5. What is stemming?**
A rule-based process that reduces words to a stem by removing prefixes/suffixes (`studies → studi`). It is fast and does not require a dictionary, so the result may not be a real word.

**6. Name stemming algorithms.**
Porter, Lancaster (Paice/Husk), Snowball (Porter2), Lovins.

**7. What is lemmatisation?**
Dictionary-based reduction to the canonical base form (lemma) using vocabulary and POS (`studies → study`, `went → go`). Accurate but slower.

**8. Biggest difference between stemming and lemmatisation?**
Stemming is heuristic and may produce non-words; lemmatisation is dictionary- and POS-aware and always produces a valid word. Stemming merges `university`/`universe` → `univers`; lemmatisation keeps them distinct.

**9. What is POS tagging? What is the tag set called?**
Assigning a grammatical category (noun, verb, adjective…) to each token, based on context. NLTK uses the Penn Treebank tag set (NN, VB, JJ, …).

**10. Why is POS tagging needed before lemmatisation?**
Because the lemma depends on the part of speech: `studying` as a noun stays `studying`, as a verb becomes `study`.

**11. What is a bag of words (BoW)?**
A document representation as a vector of word counts, ignoring grammar and word order. Simple, sparse, and loses context.

**12. Limitation of BoW?**
Ignores word order/semantics, treats all words as equally important, produces huge sparse vectors, and synonyms remain unrelated.

**13. What is TF?**
Term Frequency — how often a term occurs in a document, either a raw count or normalised by the document length (`count / total terms`).

**14. Why normalise TF by document length?**
Otherwise longer documents get larger TF values for every term, biasing similarity and model weights.

**15. What is IDF and why is it needed?**
Inverse Document Frequency = `log(N/df)`. It down-weights terms that appear in many documents, because common terms do not help distinguish documents.

**16. Write the TF-IDF formula.**
`w(t,d) = tf(t,d) × log(N / df(t))`. sklearn: `tf × (ln((1+N)/(1+df)) + 1)` followed by L2 normalisation.

**17. What happens when a term appears in all documents?**
`IDF = log(N/N) = 0`, so the term gets **zero weight** and is effectively discarded.

**18. Where is TF-IDF used?**
Text classification, spam detection, document similarity/search (with cosine similarity), keyword extraction, topic modelling, clustering news articles.

**19. What is cosine similarity and why is it used with TF-IDF?**
It measures the cosine of the angle between two document vectors: `(A·B)/(|A||B|)`. Because TF-IDF vectors are sparse and of different lengths, cosine similarity gives a length-independent measure of content overlap (1 = identical direction).

**20. What is an n-gram?**
A contiguous sequence of n tokens: unigram (1), bigram (2), trigram (3). Using bigrams captures phrases like "not good" and improves sentiment/classification at the cost of a larger vocabulary.

**21. Difference between `TfidfVectorizer` and `CountVectorizer`?**
`CountVectorizer` gives raw counts (BoW); `TfidfVectorizer` gives TF-IDF weights, which also accounts for how informative a term is across the corpus.

**22. What does `fit_transform` vs `transform` do on the vectorizer?**
`fit_transform` builds the vocabulary and IDF from the training corpus and returns the matrix; `transform` applies the learned vocabulary/IDF to new documents. Never `fit` on test data.

**23. What is the typical output of TF-IDF?**
A sparse document–term matrix of shape `(n_documents, vocabulary_size)`, where each row is a document vector.

---

## 6. Common mistakes

- Downloading only `punkt` and getting a `LookupError` for `punkt_tab` (newer NLTK).
- Calling `lemmatize("studying")` without `pos='v'` and claiming lemmatisation failed.
- Saying stemming and lemmatisation are the same thing — the viva specifically targets the difference.
- Using `log(N/df)` when `df = 0` → division by zero. Guard it (as the code does).
- Forgetting that sklearn's TF-IDF values are L2-normalised and differ from the textbook formula.
- Fitting the vectorizer on the test set → leakage and a different vocabulary.
- Removing stop words before negations matter (sentiment analysis).
