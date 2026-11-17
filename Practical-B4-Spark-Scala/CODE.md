# Practical B4 — Scala with Apache Spark

**Aim:** Write and execute a simple program in Scala using the Apache Spark framework.

**Front end:** Scala + Apache Spark
**Mode:** `spark-shell` (interactive Scala REPL provided by Spark) and standalone Spark cluster

---

## 1. Installation

### Scala

```bash
java -version                 # Java must be installed (JDK 8/11/17)

sudo apt update
sudo apt search scala
sudo apt install scala        # install from the apt repository

scala -version                # verify
```

### Apache Spark

```bash
# 1. Download (choose the version matching your Hadoop)
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz

# 2. Extract
tar -xvzf spark-3.5.1-bin-hadoop3.tgz

# 3. Move to /opt
sudo mv spark-3.5.1-bin-hadoop3 /opt/spark

# 4. Environment variables
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:/opt/spark/bin:/opt/spark/sbin' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=/usr/bin/python3' >> ~/.bashrc
source ~/.bashrc

# 5. Verify
ls -l /opt/spark
spark-shell --version
```

### Start the cluster

```bash
start-master.sh                                  # Spark master
start-workers.sh spark://localhost:7077          # one or more workers

# Web UI
# Master : http://localhost:8080
# Worker : http://localhost:8081
# App    : http://localhost:4040 (while an application runs)

spark-shell                                      # Scala REPL
```

If workers do not start:

```bash
sudo apt-get remove openssh-client openssh-server
sudo apt-get install openssh-client openssh-server
```

---

## 2. Simple Scala programs in `spark-shell`

Start `spark-shell`, then enter **paste mode** so a multi-line `object` is compiled as one unit:

```text
:paste
```

Paste the program, then press **Ctrl + D** to exit paste mode and compile it.

### (a) Even / Odd check — the main program

```scala
object EvenOddCheck {
  def main(args: Array[String]): Unit = {
    val number = 15
    if (number % 2 == 0) {
      println(s"$number is even")
    } else {
      println(s"$number is odd")
    }
  }
}
```

Run it:

```scala
EvenOddCheck.main(Array.empty[String])
```

Output:

```text
15 is odd
```

### (b) Same program with user input

```scala
import scala.io.StdIn

object AddTwoNumbers {
  def main(args: Array[String]): Unit = {
    println("Enter the first number:")
    val num1 = StdIn.readInt()
    println("Enter the second number:")
    val num2 = StdIn.readInt()
    val sum = num1 + num2
    println(s"The sum of $num1 and $num2 is $sum")
  }
}
```

```scala
AddTwoNumbers.main(Array.empty[String])
```

### (c) Factorial

```scala
object Factorial {
  def main(args: Array[String]): Unit = {
    val num = 5
    var factorial = 1
    for (i <- 1 to num) {
      factorial *= i
    }
    println(s"The factorial of $num is $factorial")
  }
}
```

### (d) Reverse a string

```scala
object ReverseString {
  def main(args: Array[String]): Unit = {
    val str = "Scala"
    val reversed = str.reverse
    println(s"The reverse of '$str' is '$reversed'")
  }
}
```

### (e) Largest element in an array

```scala
object FindLargest {
  def main(args: Array[String]): Unit = {
    val numbers = Array(10, 20, 30, 40, 50)
    val largest = numbers.max
    println(s"The largest number in the array is $largest")
  }
}
```

### (f) Positive / negative / zero

```scala
object ExCheckNumber {
  def main(args: Array[String]): Unit = {
    var number = -100
    if (number == 0) {
      println("number is zero")
    } else if (number > 0) {
      println("number is positive")
    } else {
      println("number is negative")
    }
  }
}
```

### (g) Simple calculator (pattern matching)

```scala
import scala.io.StdIn

object SimpleCalculator {
  def main(args: Array[String]): Unit = {
    println("Enter the first number:")
    val num1 = StdIn.readDouble()
    println("Enter an operator (+, -, *, /):")
    val operator = StdIn.readChar()
    println("Enter the second number:")
    val num2 = StdIn.readDouble()

    val result = operator match {
      case '+' => num1 + num2
      case '-' => num1 - num2
      case '*' => num1 * num2
      case '/' => if (num2 != 0) num1 / num2 else "undefined (division by zero)"
      case _   => "Invalid operator"
    }

    println(s"The result is: $result")
  }
}
```

---

## 3. Spark framework example — WordCount on an RDD

This is the part that actually uses Spark (not just Scala). Run it directly in `spark-shell`; `sc` is the pre-created `SparkContext`.

```scala
// Create an RDD from an in-memory collection
val lines = sc.parallelize(List(
  "hello spark",
  "hello scala",
  "spark is fast",
  "hello world"
))

// Transformation 1: split each line into words (flatten)
val words = lines.flatMap(line => line.split(" "))

// Transformation 2: map each word to (word, 1)
val pairs = words.map(word => (word, 1))

// Transformation 3: sum the 1s for each key
val counts = pairs.reduceByKey((a, b) => a + b)

// Action: bring the result to the driver and print
counts.collect().foreach(println)
```

Output (order may vary):

```text
(hello,3)
(spark,2)
(scala,1)
(is,1)
(fast,1)
(world,1)
```

Sort and display more readably:

```scala
counts.sortBy(-_._2).collect().foreach { case (w, c) => println(f"$w%-10s $c") }
```

### Reading a file from HDFS / local disk

```scala
val rdd = sc.textFile("hdfs://localhost:9000/user/hadoop/input/input.txt")
// local file: sc.textFile("file:///home/hadoop/input.txt")

val result = rdd
  .flatMap(_.toLowerCase.split("\\W+"))
  .filter(_.nonEmpty)
  .map((_, 1))
  .reduceByKey(_ + _)

result.saveAsTextFile("hdfs://localhost:9000/user/hadoop/spark_output")
```

### Submitting a packaged Scala/Spark application

```bash
spark-submit \
  --class EvenOddCheck \
  --master local[*] \
  target/scala-2.12/simple-project_2.12-1.0.jar
```

`--master local[*]` uses all local cores; `spark://localhost:7077` uses the standalone cluster.

---

## 4. Useful `spark-shell` commands

| Command | Purpose |
|---|---|
| `:paste` | enter multi-line paste mode (Ctrl+D to compile) |
| `:load file.scala` | load a script |
| `:type expr` | show the inferred type of an expression |
| `:quit` | exit the shell |
| `sc` | the `SparkContext` |
| `spark` | the `SparkSession` (Spark 2+) |
| `sc.master` | current master URL |
| `sc.defaultParallelism` | default number of partitions |
