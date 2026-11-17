# Practical B4 — Explanation & Viva Questions

## 1. What the practical is asking

"Write a simple program in **Scala** using the **Apache Spark** framework." The safest interpretation is:

1. Show the Scala installation and the Spark installation/startup.
2. Run a few simple Scala programs in `spark-shell` (even/odd, sum, factorial — the handout's list).
3. Show the Spark API in action (RDD word count).

That covers both halves of the title: **Scala** and **Spark**. The handout's own source code only shows plain Scala objects; adding the RDD example is what makes it a "Spark framework" practical.

---

## 2. Scala in the minimum words needed

Scala = **sca**lable **la**nguage. It runs on the JVM, compiles to bytecode, and mixes object-oriented and functional programming. Spark is written in Scala, which is why Scala gets the native, fastest API.

| Concept | Meaning | Example |
|---|---|---|
| `val` | immutable value (cannot be reassigned) | `val x = 10` |
| `var` | mutable variable | `var y = 10; y = 20` |
| type inference | compiler deduces the type | `val n = 15` → `Int` |
| `object` | singleton — one instance, no `new` | `object EvenOddCheck { … }` |
| `def` | method/function definition | `def add(a: Int, b: Int) = a + b` |
| `Unit` | return type meaning "no value" (like `void`) | `def main(...): Unit` |
| `Array[String]` | command-line arguments | `args(0)` |
| `s"..."` | string interpolation | `s"$number is odd"` |
| `match` | pattern matching (like a powerful switch) | `operator match { case '+' => … }` |
| `1 to n` | range | `for (i <- 1 to 5)` |
| `_` | placeholder/anonymous parameter | `reduceByKey(_ + _)` |

**Why is `main` inside an `object`?**
Scala has no static methods. An `object` is a singleton, and its `main(args: Array[String]): Unit` is the JVM entry point — equivalent to Java's `public static void main`.

**`val` vs `var`:** prefer `val`; immutability makes distributed code safer and easier to reason about. Spark itself is built around immutable RDDs.

**Why `:paste` in spark-shell?**
The REPL evaluates line by line. An `object` spanning multiple lines would be incomplete on the first line, so `:paste` collects the whole block and compiles it at once. **Ctrl + D** submits.

---

## 3. Apache Spark in the minimum words needed

**Definition:** an open-source, unified, in-memory distributed computing engine for big data — batch, streaming, SQL, machine learning and graph processing on one platform.

**Why it is faster than MapReduce:** MapReduce writes intermediate results to disk between every job. Spark keeps intermediate data **in memory** and builds a **DAG** of operations, so iterative algorithms (ML, graph) can be 10–100× faster.

### Architecture

| Component | Role |
|---|---|
| **Driver** | runs `main`, creates `SparkContext`, builds the DAG, schedules tasks |
| **SparkContext (`sc`)** | connection to the cluster; created for you in `spark-shell` |
| **Cluster Manager** | Standalone / YARN / Mesos / Kubernetes — allocates resources |
| **Worker / Executor** | JVM process on a node that runs tasks and caches data |
| **Task** | the smallest unit of work, one partition of an RDD |
| **Master** | the node that tracks workers (standalone mode) |

Ports: master RPC **7077**, master UI **8080**, worker UI **8081**, application UI **4040**.

### RDD — Resilient Distributed Dataset

An immutable, partitioned, fault-tolerant collection of records.

- **Resilient** — lost partitions are recomputed from the lineage graph.
- **Distributed** — data is split across executors.
- **Immutable** — every transformation creates a **new** RDD.

| Transformation (lazy) | Action (triggers execution) |
|---|---|
| `map`, `flatMap`, `filter` | `collect`, `count`, `first`, `take` |
| `reduceByKey`, `groupByKey` | `reduce`, `foreach` |
| `join`, `union`, `distinct` | `saveAsTextFile` |
| `sortBy`, `repartition` | `show` (DataFrame) |

**Lazy evaluation:** transformations only build the plan. Nothing runs until an action is called. This lets Spark optimise the whole pipeline (e.g. pipelining `map`+`filter` into one stage).

**Narrow vs wide transformation:** `map`/`filter` are narrow (no shuffle, data stays in the partition). `reduceByKey`/`groupByKey`/`join` are wide (require a **shuffle**, i.e. data movement across the network).

**`reduceByKey` vs `groupByKey`:** `reduceByKey` combines locally on each partition before shuffling (like a combiner), so it uses far less network and memory. Prefer `reduceByKey` unless you genuinely need all values.

---

## 4. Line-by-line explanation of the Spark word count

```scala
val lines = sc.parallelize(List("hello spark", "hello scala", ...))
```

`sc.parallelize` turns a local Scala collection into an RDD, splitting it into partitions across executors. Used for demos; in production you use `sc.textFile`.

```scala
val words = lines.flatMap(line => line.split(" "))
```

`flatMap` applies the function and **flattens** the result: each line becomes several words in one RDD, not a list of arrays. Equivalent to MapReduce's `StringTokenizer` loop.

```scala
val pairs = words.map(word => (word, 1))
```

Creates `(key, value)` tuples — exactly the mapper output `<word, 1>`.

```scala
val counts = pairs.reduceByKey((a, b) => a + b)
```

Groups by key and reduces each group with addition — the shuffle + reducer of MapReduce in one line. `_ + _` is the shorthand for `(a, b) => a + b`.

```scala
counts.collect().foreach(println)
```

`collect()` is an **action**: it runs the whole DAG, pulls every result to the driver and returns an `Array`. `foreach(println)` prints each tuple.

> Never call `collect()` on a really large RDD — it transfers everything into the driver's memory and can crash it. Use `take(n)` or `saveAsTextFile`.

`sortBy(-_._2)` sorts by the second element of the tuple (the count) in descending order (`-` negates). `_._2` means "second field of the tuple".

---

## 5. Journal questions from the handout (with answers)

**Q. Write down the steps to install Scala.**
1. Verify Java: `java -version`.
2. Update packages: `sudo apt update`.
3. Search the package: `sudo apt search scala`.
4. Install: `sudo apt install scala`.
5. Verify: `scala -version`.
6. (Optional) Set `SCALA_HOME` and add `$SCALA_HOME/bin` to `PATH`; or install via `sdkman`/`cs setup` for a specific version.

**Related: steps to install Apache Spark.**
1. Download the binary matching your Hadoop version (`spark-3.x-bin-hadoop3.tgz`).
2. Extract with `tar -xvzf`.
3. Move to `/opt/spark`.
4. Export `SPARK_HOME` and add `$SPARK_HOME/bin:$SPARK_HOME/sbin` to `PATH` in `~/.bashrc`.
5. `source ~/.bashrc`, verify with `spark-shell --version` or `ls -l /opt/spark`.
6. Start the cluster: `start-master.sh`, `start-workers.sh spark://localhost:7077`.
7. Open `http://localhost:8080` and run `spark-shell`.

---

## 6. Viva questions the external examiner may ask

**1. What is Scala? Why does Spark use it?**
A JVM language combining object-oriented and functional programming with concise syntax, immutability and pattern matching. Spark is written in Scala, so the Scala API is native, type-safe and fastest; Scala's functional style maps naturally onto distributed transformations.

**2. Difference between `val` and `var`?**
`val` is immutable (a constant reference, cannot be reassigned); `var` is mutable. Immutability is preferred, especially in distributed code, because it avoids shared-state bugs.

**3. What is an `object` in Scala?**
A singleton instance of a class. It replaces Java's statics. `object X { def main(...) }` is the program entry point.

**4. Difference between Scala `object` and `class`?**
A `class` needs `new` and can have many instances; an `object` has exactly one instance created by the compiler and is used for utilities, constants and the `main` method.

**5. What is `Unit`?**
Scala's equivalent of `void` — a method that returns nothing meaningful. `main` returns `Unit`.

**6. What is string interpolation in Scala?**
`s"Value = $x"` substitutes variables; `${expr}` substitutes expressions; `f"$x%.2f"` is type-safe formatting. It replaces Java's string concatenation.

**7. What is pattern matching?**
A `match` expression that compares a value against patterns (`case '+' => …`, `case _ => …`), far more powerful than a switch — it can destructure types and collections. Its result is an expression, so it can be assigned.

**8. What is Apache Spark?**
An open-source, distributed, in-memory data processing engine for batch, streaming, SQL, ML and graph workloads. It runs on YARN, Mesos, Kubernetes or its own standalone cluster.

**9. How is Spark faster than MapReduce?**
In-memory computation, DAG-based execution (fewer stages), lazy evaluation with whole-pipeline optimisation, and no mandatory disk write between stages. For iterative ML, Spark avoids re-reading disk on every iteration.

**10. What is an RDD?**
Resilient Distributed Dataset — an immutable, partitioned collection distributed across executors, with lineage-based fault recovery. The fundamental (low-level) Spark abstraction.

**11. What is the difference between a transformation and an action?**
Transformations are lazy and return a new RDD (`map`, `filter`, `reduceByKey`). Actions trigger execution and return a value or write output (`collect`, `count`, `saveAsTextFile`).

**12. What does lazy evaluation mean? Why is it useful?**
Nothing executes until an action is called. Spark can then optimise the full chain (combine narrow transformations, avoid unnecessary shuffles) instead of executing each step eagerly.

**13. What is a DAG in Spark?**
A Directed Acyclic Graph of transformations from input to output. The scheduler splits it into **stages** at wide transformations (shuffle boundaries) and submits tasks per partition.

**14. What is a shuffle? Why is it expensive?**
The redistribution of data across partitions so that all records with the same key land together. It requires network I/O, disk I/O and serialisation — usually the dominant cost.

**15. `reduceByKey` vs `groupByKey`?**
`reduceByKey` performs local combining before the shuffle (map-side aggregation) and returns one value per key. `groupByKey` shuffles all values and groups them, which is memory-heavy. Prefer `reduceByKey`.

**16. What is the difference between `map` and `flatMap`?**
`map` returns one element per input; `flatMap` returns 0, 1 or many and flattens the results (used to split lines into words).

**17. What is `collect()`? When should you avoid it?**
An action that returns the entire RDD to the driver as an Array. Avoid it on large datasets — it can cause driver out-of-memory. Use `take(n)`, `first()`, or write to storage.

**18. What is the difference between the driver and an executor?**
The driver runs the user's `main`, builds the DAG and schedules tasks. Executors run tasks and store cached/partitioned data on the workers.

**19. What is `spark-shell`?**
Spark's interactive Scala REPL, pre-configured with a `SparkContext` (`sc`) and `SparkSession` (`spark`). It is for exploration; production jobs are packaged and run with `spark-submit`.

**20. Difference between `local[*]` and `spark://localhost:7077`?**
`local[*]` runs the driver and all "workers" as threads inside one JVM (no cluster). `spark://host:7077` uses the standalone cluster's master, with separate worker JVMs.

**21. What is the difference between RDD, DataFrame and Dataset?**
RDD = low-level, untyped, full control, no optimiser. DataFrame = schema-based, columnar, Catalyst-optimised (fastest for SQL-like work). Dataset = typed + Catalyst (Scala/Java only). Modern Spark code prefers DataFrames/Datasets; RDDs are used for fine-grained control.

**22. What is a Partition in Spark?**
A logical chunk of an RDD/DataFrame that one task processes. More partitions = more parallelism, but too many small partitions add scheduling overhead.

**23. What is caching/persistence? Why is it useful?**
`rdd.cache()` or `.persist()` keeps data in memory/disk across actions. It is essential for iterative algorithms that reuse the same dataset.

**24. What is lineage and how does Spark recover from failure?**
Each RDD remembers how it was derived from its parents. If a partition is lost, Spark recomputes only that partition from the lineage rather than re-running the whole job.

**25. What is `spark-submit`?**
The command-line launcher for Spark applications. It sets the class, master, deploy mode (client/cluster), memory/cores and dependencies.

**26. What is the Spark ecosystem?**
Spark Core, Spark SQL, Spark Streaming / Structured Streaming, MLlib (machine learning), GraphX (graph). External integrations: Kafka, Hive, HDFS, JDBC.

---

## 7. Common mistakes

- Typing a multi-line `object` in `spark-shell` **without** `:paste` — each line fails to compile separately.
- Forgetting **Ctrl + D** to leave paste mode.
- Using `collect()` on a large RDD.
- Confusing Scala `val` (immutable) with `var`.
- Saying "Spark is faster because it uses RAM" without mentioning DAG/lazy evaluation/avoided disk I/O.
- Claiming `reduceByKey` and `groupByKey` are equivalent in performance.
- Forgetting that `start-master.sh` / `start-workers.sh` are only for standalone mode (YARN uses `yarn`).
- Not mentioning the Spark UI ports (8080 master, 4040 application).
