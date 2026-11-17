# Practical B2 — Explanation & Viva Questions

## 1. What the practical is asking

"Design a **distributed application** using MapReduce which processes a **log file of a system**." There is no prescribed output, so you choose a meaningful aggregation. The safest choice (and the one used here) is **count of log entries per severity level** — the same map → shuffle → sum pattern as WordCount, applied to a structured log line.

A second accepted variant is **requests per IP address**, included in `CODE.md`.

---

## 2. Why log analysis is a MapReduce problem

Server logs are:

- **huge** — GBs per day for a busy service,
- **append-only and line-oriented** — each line is independent,
- **embarrassingly parallel** — no line needs any other line to be parsed.

That is exactly the MapReduce profile: split the file by block, parse each line independently in the mapper, aggregate per key in the reducer.

Real-world analyses: errors per hour, requests per IP, top URLs, HTTP status distribution, average response time per endpoint, failed login attempts per user.

---

## 3. Line-by-line explanation

### Line format

```
2024-01-01 10:00:12 ERROR Database connection failed
└─ date ─┘ └─ time ─┘ └level┘ └──────── message ────────┘
```

`parts[0]` = date, `parts[1]` = time, `parts[2]` = severity, rest = message.

### Mapper

```java
String line = value.toString().trim();
if (line.isEmpty()) return;
```

Defensive: skip blank lines. Real log files contain them.

```java
String[] parts = line.split("\\s+");
if (parts.length < 3) return;
```

`\\s+` splits on one or more whitespace characters (spaces/tabs). Lines shorter than 3 tokens cannot contain a level, so they are skipped. **Skipping malformed input is essential in production** — one bad line must not crash the job.

```java
String level = parts[2].toUpperCase();
if (level.equals("INFO") || level.equals("WARN")
        || level.equals("ERROR") || level.equals("DEBUG")) {
    logLevel.set(level);
    context.write(logLevel, one);
}
```

Only known severities are emitted. This is a **whitelist filter** inside the map phase — cheaper than filtering later because the reducer never sees noise.

> Note: `switch` on a String would be neater, but the `if` chain is portable and easy to explain. `Text` objects must be compared with `.equals`, never `==`.

### Reducer

```java
int count = 0;
for (IntWritable val : values) count += val.get();
```

Hadoop has already grouped every `1` belonging to `"ERROR"` into the `Iterable`. Summing them gives the count. This is identical in structure to WordCount — which is the point to make: **the pattern generalises**.

### Combiner

```java
job.setCombinerClass(LogReducer.class);
```

Each mapper locally sums its own `ERROR` counts before sending them to the reducer. Network traffic drops from one record per line to one record per (mapper, level). Safe because addition is associative and commutative.

### Driver

`job.setJarByClass`, input/output paths, `waitForCompletion`. Same driver skeleton as every MapReduce job.

---

## 4. Reading the output

```
DEBUG   2
ERROR   3
INFO    5
WARN    2
```

Sorted alphabetically (Hadoop sorts keys). Interpretation to write in the journal:

- 12 log entries processed; no parse failures.
- 25 % are `ERROR` (3/12) — database connection failures and a payment gateway timeout, which are the operationally important ones.
- 2 warnings (disk usage, slow response) are early indicators of the two errors.
- A production dashboard would alert when the `ERROR` count crosses a threshold per hour.

---

## 5. The IP-count variant (why it is also valid)

`LogIPCount.map` parses the **first token** of an access-log line and validates it with the regex `\d{1,3}(\.\d{1,3}){3}` before emitting `(ip, 1)`. The shape is identical — only the key extraction differs. This demonstrates the general principle: **in MapReduce the mapper decides "what is the key", and the reducer counts/aggregates per key**.

Be ready to explain the regex: `\d{1,3}` = 1 to 3 digits, `(\.\d{1,3}){3}` = three repetitions of "dot + 1–3 digits", giving `a.b.c.d`. `String.matches` must match the **whole** string, so no anchors are needed.

---

## 6. Journal questions from the handout (with answers)

**Q1. Explain data structures of big data.**
Big data does not fit relational rows/columns comfortably, so several models are used:
- **Key–value** (HBase, Redis, MapReduce intermediate data) — simplest, good for lookups.
- **Document** (MongoDB, CouchDB) — JSON-like nested records.
- **Column-family / wide-column** (HBase, Cassandra) — sparse, column-oriented, good for writes and scans.
- **Graph** (Neo4j, Giraph) — nodes/edges for social networks, paths.
- **Object storage** (S3, HDFS blocks) — files as blobs for batch processing.
The **5 Vs** — Volume, Velocity, Variety, Veracity, Value — explain why these structures are needed.

**Q2. Explain applications of Hadoop.**
Banking and fraud detection (transaction scans), retail recommendation and clickstream analysis, telecom call-record analysis, healthcare records, log/ETL processing, search indexing (web crawl + inverted index), social network analysis, IoT sensor data, data warehousing offload (with Hive), image/video batch processing, backup and archival.

**Q3. Explain the Hadoop ecosystem.**
| Component | Role |
|---|---|
| HDFS | distributed storage |
| YARN | resource management and scheduling |
| MapReduce | batch processing |
| Hive | SQL-like data warehouse on HDFS |
| Pig | scripting (Pig Latin) for ETL |
| HBase | NoSQL column store on HDFS |
| Sqoop | bulk import/export between RDBMS and HDFS |
| Flume | streaming log/event ingestion |
| Oozie | workflow scheduler |
| ZooKeeper | distributed coordination |
| Spark | in-memory fast processing (batch + streaming + ML) |
| Kafka | distributed publish-subscribe event streaming |
| Mahout / MLlib | machine learning libraries |

---

## 7. Viva questions the external examiner may ask

**1. What is a distributed application?**
An application whose computation and data are spread across multiple machines that cooperate over a network, so it can scale beyond one machine's CPU/memory/disk and survive individual node failures.

**2. How is this log-processing job "distributed"?**
The input file is split into HDFS blocks; each split is processed by a mapper on the node that stores it (data locality); intermediate pairs are shuffled by key to reducers, which may run on other nodes; output is written back to HDFS with replication.

**3. Why is the mapper's output key `Text` and not `IntWritable`?**
The key must be the thing we group by — here the log level, which is text. Values are the counts (`IntWritable`). Key type and value type are independent.

**4. What happens to malformed log lines?**
They are skipped (`parts.length < 3`). In production you would also increment a custom counter to monitor how many were dropped.

**5. What is a Hadoop counter?**
A global, distributed counter maintained by the framework (e.g. `MAP_INPUT_RECORDS`, `REDUCE_OUTPUT_RECORDS`). You can define your own with `context.getCounter("LogStats", "MALFORMED").increment(1)` and read them after the job.

**6. Why is the combiner safe here but not for an average?**
Summation is associative and commutative, so partial sums can be summed again. An average cannot: `avg(avg(1,2), 3) = 1.75 ≠ 2`.

**7. What if two mappers emit the same level?**
The partitioner sends both to the same reducer (same key → same hash), and Hadoop merges the values into one `Iterable` before calling `reduce` once.

**8. What if we use 2 reducers?**
Keys are distributed by hash: e.g. INFO/ERROR might go to reducer 0 and WARN/DEBUG to reducer 1, producing `part-r-00000` and `part-r-00001`. The totals are still correct, but split across files.

**9. How would you find the top 10 IPs by request count?**
Job 1: count per IP (this program). Job 2: read the counts, use a reducer that keeps a sorted top-N, or sort by value in descending order (a secondary sort / custom comparator). In Spark it is simply `reduceByKey(_+_).sortBy(-_._2).take(10)`.

**10. How would you process logs arriving continuously?**
MapReduce is batch. Use **Kafka + Spark Streaming / Flink** for real-time ingestion and windowed aggregation, or Flume to land files in HDFS and run periodic MapReduce/Spark batches.

**11. Why is log data suitable for HDFS?**
It is large, append-only, written once and read many times — exactly HDFS's design goal. Small random updates would be a poor fit.

**12. What is data locality?**
Scheduling a mapper on the same node that holds the input block, so the data is read from local disk instead of over the network. It is a major reason MapReduce scales.

**13. What is the difference between `Text` and `String` in Hadoop?**
`Text` stores UTF-8 bytes, is mutable and reusable, and implements `WritableComparable` for serialisation and sorting — required for keys and values in MapReduce. `String` is immutable and Java-serialised, which is inefficient.

**14. What does `split("\\s+")` do?**
Splits the string on one or more whitespace characters. The backslash is escaped in the Java string literal, so the regex engine sees `\s+`.

**15. Can the same job read multiple input files?**
Yes. `FileInputFormat.addInputPath` can be called several times, or you can pass a directory. Each file/split is processed by a mapper.

**16. What is the role of `Configuration`?**
It carries the job's settings (HDFS address, serialisation, custom properties). `new Configuration()` loads `core-site.xml`, `hdfs-site.xml`, `mapred-site.xml` and `yarn-site.xml` from the classpath.

**17. What is speculative execution?**
If a task runs much slower than others (a "straggler"), Hadoop launches a duplicate attempt on another node and takes whichever finishes first. Controlled by `mapreduce.map.speculative` / `reduce.speculative`.

**18. What are the main failure points in a MapReduce job?**
Mapper/reducer crash (task retried), DataNode failure (block re-read from a replica), NodeManager failure (task rescheduled), ApplicationMaster failure (attempt restarted by YARN), and bad input records (should be filtered).

**19. How would you count errors per hour instead of per level?**
Emit a composite key `(date + " " + hour + " " + level)` from the mapper, then the reducers count per hour–level combination. Or emit `(hour, level)` using a custom `WritableComparable`.

**20. Why must you remove the output directory before re-running?**
Hadoop validates that the output path does not exist to avoid overwriting previous results. Use `hadoop fs -rm -r /output` or a timestamped path.

---

## 8. Common mistakes

- Assuming column positions (`parts[2]`) without checking the log format — always state the assumed format in the journal.
- Crashing the whole job on one malformed line instead of skipping it.
- Using `==` to compare `Text`/`String` values.
- Calling a combiner for a non-associative metric.
- Forgetting that output must not already exist.
- Using the default 1 reducer and then claiming the job is fully parallel at the reduce stage.
- Not showing the exact expected output in the journal.
