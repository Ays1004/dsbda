# Practical B2 — MapReduce Log File Analysis

**Aim:** Design a distributed application using MapReduce that processes a system log file and produces a summary (count of log entries per severity level).

**Front end:** Java + Hadoop
**Input:** `logfile.txt` — one log entry per line

---

## 1. `LogAnalyzer.java`

```java
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class LogAnalyzer {

    // ---------------- MAPPER ----------------
    // Input : <offset, "2024-01-01 10:00:12 ERROR Database connection failed">
    // Output: <"ERROR", 1>
    public static class LogMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text logLevel = new Text();

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().trim();
            if (line.isEmpty()) {
                return;                       // skip blank lines
            }

            // Format assumed:  DATE TIME LEVEL MESSAGE
            String[] parts = line.split("\\s+");
            if (parts.length < 3) {
                return;                       // malformed line
            }

            String level = parts[2].toUpperCase();

            // Keep only recognised severities
            if (level.equals("INFO") || level.equals("WARN")
                    || level.equals("ERROR") || level.equals("DEBUG")) {
                logLevel.set(level);
                context.write(logLevel, one);
            }
        }
    }

    // ---------------- REDUCER ----------------
    // Input : <"ERROR", [1,1,1,...]>
    // Output: <"ERROR", 3>
    public static class LogReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable total = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int count = 0;
            for (IntWritable val : values) {
                count += val.get();
            }
            total.set(count);
            context.write(key, total);
        }
    }

    // ---------------- DRIVER ----------------
    public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println("Usage: LogAnalyzer <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "log level analysis");

        job.setJarByClass(LogAnalyzer.class);
        job.setMapperClass(LogMapper.class);
        job.setCombinerClass(LogReducer.class);   // sum is associative -> safe
        job.setReducerClass(LogReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

---

## 2. Input file — `logfile.txt`

```text
2024-01-01 10:00:01 INFO User login success
2024-01-01 10:00:05 INFO User opened dashboard
2024-01-01 10:00:09 WARN Disk usage above 80 percent
2024-01-01 10:00:12 ERROR Database connection failed
2024-01-01 10:00:18 INFO Report generated
2024-01-01 10:00:23 DEBUG Cache refreshed
2024-01-01 10:00:31 ERROR Payment gateway timeout
2024-01-01 10:00:35 INFO User logout
2024-01-01 10:00:41 WARN Slow response from server
2024-01-01 10:00:47 ERROR Database connection failed
2024-01-01 10:00:52 INFO User login success
2024-01-01 10:00:58 DEBUG Session cleaned up
```

---

## 3. Compile and run

```bash
# Compile against Hadoop libraries
javac -classpath `hadoop classpath` -d . LogAnalyzer.java

# Package
jar cvf LogAnalyzer.jar *.class

# Upload input to HDFS
hadoop fs -mkdir -p /user/hadoop/log_input
hadoop fs -put logfile.txt /user/hadoop/log_input/
hadoop fs -ls /user/hadoop/log_input

# Run the job
hadoop jar LogAnalyzer.jar LogAnalyzer \
    /user/hadoop/log_input/logfile.txt /user/hadoop/log_output

# Display the result
hadoop fs -cat /user/hadoop/log_output/part-r-00000
```

---

## 4. Expected output

```text
DEBUG   2
ERROR   3
INFO    5
WARN    2
```

---

## 5. Variant — count requests per IP address (access log)

Save as `LogIPCount.java` (separate file, separate job).

```java
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class LogIPCount {

    // Input line: 192.168.1.10 - - [01/Jan/2024:10:00:01] "GET /index.html" 200
    // Output    : <ip, 1>
    public static class IPMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text ip = new Text();

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().trim();
            if (line.isEmpty()) return;

            String firstToken = line.split("\\s+")[0];

            // basic IPv4 validation, skip comment/header lines
            if (firstToken.matches("\\d{1,3}(\\.\\d{1,3}){3}")) {
                ip.set(firstToken);
                context.write(ip, one);
            }
        }
    }

    public static class IPReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable total = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int count = 0;
            for (IntWritable v : values) count += v.get();
            total.set(count);
            context.write(key, total);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "requests per ip");

        job.setJarByClass(LogIPCount.class);
        job.setMapperClass(IPMapper.class);
        job.setCombinerClass(IPReducer.class);
        job.setReducerClass(IPReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

```bash
javac -classpath `hadoop classpath` -d . LogIPCount.java
jar cvf LogIPCount.jar *.class
hadoop jar LogIPCount.jar LogIPCount /user/hadoop/log_input /user/hadoop/ip_output
hadoop fs -cat /user/hadoop/ip_output/part-r-00000
```
