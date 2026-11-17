# Practical B1 — Explanation & Viva Questions

## 1. What the practical is asking

Write a Java **WordCount** program, compile it against the Hadoop jars, run it as a MapReduce job on HDFS, and show the output in `part-r-00000`.

The examiner typically checks: the three classes (Mapper, Reducer, Driver), the `<key, value>` types, and Hadoop/HDFS terminology.

---

## 2. Hadoop in one paragraph

Apache Hadoop is an open-source framework for **distributed storage and distributed processing** of very large datasets on clusters of commodity hardware. It has four modules:

| Module | Purpose |
|---|---|
| **HDFS** | Hadoop Distributed File System — stores data across machines |
| **YARN** | Yet Another Resource Negotiator — manages CPU/memory and schedules jobs |
| **MapReduce** | Programming model for parallel batch processing |
| **Common** | Shared utilities/libraries used by the other three |

**Core idea:** move computation to the data instead of moving data to the computation.

---

## 3. HDFS

- Data is split into **blocks** (default 128 MB in Hadoop 2/3, 64 MB in Hadoop 1).
- Each block is replicated (default **replication factor = 3**) on different DataNodes.
- **NameNode** — master; stores the metadata (namespace, block locations).
- **DataNode** — slave; stores the actual blocks and serves read/write requests.
- **SecondaryNameNode** — periodically merges the edit log into the fsimage; it is **not** a backup NameNode.
- Default port: NameNode web UI **9870** (Hadoop 3), RPC **9000**.

Useful commands:

```bash
hadoop fs -mkdir /input
hadoop fs -put file.txt /input
hadoop fs -ls /input
hadoop fs -cat /output/part-r-00000
hadoop fs -get /output/part-r-00000
hadoop fs -rm -r /output
hadoop fs -df -h            # free space
```

---

## 4. MapReduce — the flow

```
Input splits
    |
    v
RecordReader  ->  <k1, v1>
    |
    v
 MAPPER  (your map function)  ->  <k2, v2>       [map stage]
    |
    v
 Combiner (optional, local reduce)  ->  <k2, v2>
    |
    v
 Partitioner decides which reducer gets which key
    |
    v
 Shuffle & Sort  ->  <k2, [v2, v2, ...]>          [shuffle stage]
    |
    v
 REDUCER  (your reduce function)  ->  <k3, v3>   [reduce stage]
    |
    v
OutputFormat  ->  part-r-00000
```

**WordCount mapping:** input line → split into words → emit `(word, 1)` → shuffle groups all 1s for the same word → reducer sums them → `(word, count)`.

### The three classes

| Class | Extends | Input | Output |
|---|---|---|---|
| `TokenizerMapper` | `Mapper<Object, Text, Text, IntWritable>` | `<offset, line>` | `<word, 1>` |
| `IntSumReducer` | `Reducer<Text, IntWritable, Text, IntWritable>` | `<word, [1,1,…]>` | `<word, sum>` |
| `WordCount` (driver) | — | `args[0]`, `args[1]` | job configuration + `waitForCompletion` |

### Why `Text` and `IntWritable` instead of `String` and `int`?

Hadoop serialises data to move it between nodes. Java's `Serializable` is bulky, so Hadoop defines the `Writable` interface:

- `IntWritable`, `LongWritable`, `FloatWritable`, `DoubleWritable`, `BooleanWritable`, `Text` (UTF-8 string), `NullWritable`.
- Keys must also be **sortable**, so they implement `WritableComparable`.
- `Text` is used instead of `String` because it is mutable and stores UTF-8 bytes compactly.

### Combiner

```java
job.setCombinerClass(IntSumReducer.class);
```

A **mini-reducer that runs on the mapper's local output** before the shuffle. It turns 1000 `(meow,1)` pairs into `(meow,1000)` on the same node, drastically reducing network traffic. It is optional and only safe when the operation is **associative and commutative** (sum, max, min — not average).

### Partitioner

Default `HashPartitioner` computes `hash(key) % numReducers`. All values for the same key always go to the same reducer. With 2 reducers the output is split into `part-r-00000` and `part-r-00001`.

### Number of mappers and reducers

- One mapper per **input split** (usually one per HDFS block).
- Reducers are set explicitly (`job.setNumReduceTasks(n)`); default is 1. The number of output part files equals the number of reducers.

---

## 5. Line-by-line explanation of the code

### Mapper

```java
public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>
```

The four type parameters are `<input key, input value, output key, output value>`. The input key is the byte offset of the line (rarely used, hence `Object`).

```java
StringTokenizer itr = new StringTokenizer(value.toString());
```

`value` is one line of the file. `StringTokenizer` splits on whitespace by default.

```java
word.set(itr.nextToken().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""));
```

`Text` is reused (not re-created) for efficiency — a standard Hadoop optimisation. Lowercasing makes `Hello` and `hello` the same key; the regex removes punctuation so `code.` becomes `code`.

```java
context.write(word, one);
```

Emits the intermediate pair. `context` is how a Mapper/Reducer writes output.

### Reducer

```java
public void reduce(Text key, Iterable<IntWritable> values, Context context)
```

Hadoop groups all values for the same key into one `Iterable` — your code never sees the shuffle.

```java
for (IntWritable val : values) sum += val.get();
result.set(sum);
context.write(key, result);
```

Sums the 1s. `IntWritable` is mutable, so `result` is reused.

### Driver

```java
Job job = Job.getInstance(conf, "word count");
job.setJarByClass(WordCount.class);
```

The job needs to know which jar contains the classes so it can ship them to the task nodes.

```java
FileInputFormat.addInputPath(job, new Path(args[0]));
FileOutputFormat.setOutputPath(job, new Path(args[1]));
```

Input may be a file or a directory; output **must not exist**, because Hadoop refuses to overwrite (prevents accidental data loss).

```java
System.exit(job.waitForCompletion(true) ? 0 : 1);
```

`waitForCompletion(true)` blocks until the job finishes, prints progress, and returns `true` on success. Exit code 0 = success, 1 = failure.

---

## 6. Journal questions from the handout (with answers)

**Q1. Explain Hadoop daemons.**
Daemons are background Java processes that make up a Hadoop cluster.

| Daemon | Runs on | Role |
|---|---|---|
| **NameNode** | master | metadata, namespace, block mapping |
| **DataNode** | slave | stores blocks, serves reads/writes |
| **SecondaryNameNode** | master (separate host usually) | checkpoints/merges edit logs; not a standby |
| **ResourceManager** | master | allocates cluster resources, schedules applications (YARN) |
| **NodeManager** | slave | manages containers and resources on one node |
| **JobHistoryServer** | master | stores/ serves completed MapReduce job history |
| **ApplicationMaster** | per application | negotiates resources and monitors one job's tasks |

**Q2. Explain HDFS.**
Hadoop Distributed File System — a distributed, fault-tolerant file system. It splits files into fixed-size blocks (128 MB), replicates each block (default 3×) across DataNodes, and keeps metadata on the NameNode. It follows a master–slave architecture, is designed for **write-once-read-many** workloads, and tolerates node failure through replication. It favours throughput over low latency.

**Q3. Explain the MapReduce framework.**
A distributed processing model with three stages: **map** (process each input split in parallel and emit intermediate key–value pairs), **shuffle & sort** (group and sort all values by key, partition them to reducers), and **reduce** (aggregate the grouped values into final output). Developers only write the mapper and reducer; Hadoop handles splitting, scheduling, data movement, fault tolerance and retries. It scales horizontally by adding nodes.

---

## 7. Viva questions the external examiner may ask

**1. What is Hadoop?**
An open-source Apache framework for distributed storage (HDFS) and distributed processing (MapReduce/YARN) of big data on commodity hardware. It provides scalability, fault tolerance and cost-effective storage.

**2. What are the core components of Hadoop?**
HDFS, YARN, MapReduce and Hadoop Common. (The wider ecosystem includes Hive, Pig, HBase, Sqoop, Flume, Spark, Oozie, ZooKeeper.)

**3. What is the difference between HDFS and a normal file system?**
HDFS is distributed and fault-tolerant: blocks are spread over many machines and replicated; the NameNode holds metadata. It is optimised for large sequential writes/reads, not for random access or many small files.

**4. What is the default block size and replication factor?**
128 MB block, replication 3 (Hadoop 2/3). Hadoop 1 used 64 MB.

**5. Why is a large block size used?**
To reduce the number of blocks, metadata size and seek overhead, making sequential throughput the dominant cost. It also allows one mapper per block to process large contiguous chunks.

**6. What happens if a DataNode fails?**
The NameNode detects the missing heartbeats, marks the node dead, and re-replicates its blocks from other replicas to maintain the replication factor.

**7. What is the difference between NameNode and SecondaryNameNode?**
NameNode is the active master holding metadata. The SecondaryNameNode only periodically merges the edit log with the fsimage to keep the NameNode's startup fast; it is **not** a hot standby and cannot serve requests.

**8. What is a "split" in MapReduce?**
A logical chunk of the input (usually aligned with an HDFS block) processed by one mapper. It is not a physical copy of the data.

**9. Difference between map and reduce?**
Map is parallel per-split transformation (filter, parse, emit) with no knowledge of other records. Reduce is per-key aggregation after the shuffle, where all values of a key are available together.

**10. What is the shuffle?**
The phase between map and reduce where intermediate pairs are partitioned by key, sorted, and transferred across the network so that all values for a key reach the same reducer. It is the most expensive part of a MapReduce job.

**11. What is a combiner and how is it different from a reducer?**
A combiner is a local, optional "mini-reducer" run on the map output to reduce network traffic. Unlike the reducer, it may run zero or more times, and it must operate on an associative/commutative function. A reducer gets the full key group across all mappers.

**12. Why can't a combiner be used for average?**
Because averaging is not associative across partial results: `avg(avg(a,b), c) ≠ avg(a,b,c)`. You must emit `(sum, count)` pairs and divide in the reducer.

**13. What is a partitioner?**
The component that decides which reducer receives a given key: `hash(key) % numReducers`. It guarantees all values of one key go to one reducer.

**14. Why must the output directory not exist?**
To prevent silently overwriting results of a previous run. Delete it first (`hadoop fs -rm -r`) or use a new path.

**15. What is `part-r-00000`?**
The output file written by reducer 0 (r = reduce). With `n` reducers you get `part-r-00000 … part-r-0000(n-1)`. Map-only jobs write `part-m-00000`.

**16. What is the difference between standalone, pseudo-distributed and fully distributed mode?**
Standalone: everything in one JVM, local file system, no HDFS daemons. Pseudo-distributed: all daemons on one machine, HDFS + YARN active (used in the lab). Fully distributed: daemons spread over multiple machines.

**17. What is `hadoop classpath` used for?**
It prints all Hadoop jar paths, so `javac -classpath` can compile your program against the Hadoop API. `hadoop jar` then runs the packaged jar with the cluster's configuration.

**18. Why `Text` and `IntWritable` rather than `String` and `int`?**
Custom `Writable` types provide fast, compact serialisation for network transfer and disk, and keys implement `WritableComparable` so Hadoop can sort them.

**19. What is `job.waitForCompletion(true)`?**
It submits the job and waits; `true` enables verbose progress. It returns `true` if the job succeeded, which is mapped to exit code 0.

**20. Can WordCount be done without a reducer?**
Yes. A **map-only job** with `setNumReduceTasks(0)` writes mapper output directly (`part-m-00000`), but then identical words are not aggregated across splits.

**21. What are the limitations of Hadoop MapReduce?**
High latency from disk-based I/O and shuffle; not suitable for iterative algorithms (ML) or interactive queries; poor for small files; complex to develop compared with Spark. Spark keeps data in memory and is typically 10–100× faster for iterative jobs.

**22. What is the Hadoop ecosystem?**
HDFS (storage), YARN (resource management), MapReduce (batch), Hive (SQL-like warehouse), Pig (scripting), HBase (NoSQL column store), Sqoop (RDBMS import/export), Flume (log ingestion), Oozie (workflow), ZooKeeper (coordination), Spark (in-memory processing), Kafka (streaming).

---

## 8. Common mistakes

- Running the job with an output directory that already exists.
- Forgetting `-classpath \`hadoop classpath\`` when compiling.
- Using `String` instead of `Text` in the Mapper/Reducer signatures — it will not compile.
- Comparing `Text` objects with `==` instead of `.toString().equals(...)`.
- Using a combiner for a non-associative operation (average, median).
- Expecting `part-r-00000` when the job used 0 reducers (then it is `part-m-00000`).
- Not starting HDFS (`start-all.sh`) before using `hadoop fs` / `hadoop jar`.
