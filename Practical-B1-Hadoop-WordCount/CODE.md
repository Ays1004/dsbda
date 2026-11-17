# Practical B1 — Hadoop WordCount (MapReduce)

**Aim:** Write a Java program for a simple WordCount application that counts the number of occurrences of each word in an input file using the Hadoop MapReduce framework on a local standalone setup.

**Front end:** Java + Hadoop
**Input:** `input.txt` (any text file; the handout uses `fruits.txt`)
**Output:** `part-r-00000` — one line per word with its count

---

## 1. `WordCount.java`

```java
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    // ---------------- MAPPER ----------------
    // Input : <byte offset, one line of text>
    // Output: <word, 1> for every word in the line
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                // lowercase + remove punctuation so "Hello" and "hello," match
                word.set(itr.nextToken().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""));
                if (!word.toString().isEmpty()) {
                    context.write(word, one);
                }
            }
        }
    }

    // ---------------- REDUCER ----------------
    // Input : <word, [1,1,1,...]>
    // Output: <word, total count>
    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    // ---------------- DRIVER (main) ----------------
    public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println("Usage: WordCount <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");

        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);   // local pre-aggregation, optional
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

---

## 2. Input file — `input.txt`

```text
This is a sample code.
All the way from KSKA Git.
Hello world
Meow meow meow meow
```

---

## 3. Compile and run

```bash
# 1. Compile (hadoop classpath supplies all Hadoop jars)
javac -classpath `hadoop classpath` -d . WordCount.java

# 2. Package into a jar
jar cvf WordCount.jar *.class
jar tf WordCount.jar          # verify the classes are inside

# 3. Create directories in HDFS
hadoop fs -mkdir -p /user/hadoop/input
hadoop fs -mkdir -p /user/hadoop/output

# 4. Upload the input file to HDFS
hadoop fs -put input.txt /user/hadoop/input/
hadoop fs -ls /user/hadoop/input

# 5. Run the job (output directory must NOT already exist)
hadoop jar WordCount.jar WordCount /user/hadoop/input/input.txt /user/hadoop/output

# 6. View the result
hadoop fs -ls /user/hadoop/output
hadoop fs -cat /user/hadoop/output/part-r-00000

# 7. Copy the result back to the local file system
hadoop fs -get /user/hadoop/output/part-r-00000
```

> If a previous output directory exists: `hadoop fs -rm -r /user/hadoop/output` before re-running.

---

## 4. Expected output (`part-r-00000`)

```text
all     1
code    1
from    1
git     1
hello   1
is      1
kska    1
meow    4
sample  1
the     1
this    1
way     1
world   1
```

The output is sorted alphabetically because MapReduce sorts the keys before the reduce phase.

---

## 5. Minimal Hadoop start/stop commands (pseudo-distributed)

```bash
# Format the NameNode once, before the first start
hdfs namenode -format

# Start everything (HDFS + YARN)
start-all.sh            # or: start-dfs.sh && start-yarn.sh

# Check the daemons
jps                     # NameNode, DataNode, SecondaryNameNode,
                        # ResourceManager, NodeManager, Jps

# Web UIs
# NameNode : http://localhost:9870
# YARN     : http://localhost:8088

# Stop everything
stop-all.sh
```
