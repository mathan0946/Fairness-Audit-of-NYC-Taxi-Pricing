# Using Scala Spark for NYC Taxi Fairness Audit

## 🎯 Overview

This guide shows you how to use **Apache Spark with Scala** for the taxi fairness audit project. Scala is Spark's native language and offers better performance than PySpark for complex operations.

## 📋 Prerequisites

- Apache Spark 3.5.1 installed locally
- Scala 2.12 or 2.13
- `SPARK_HOME` environment variable set
- Spark binaries in PATH

## 🚀 Quick Start

### Method 1: Using Spark Shell (Interactive)

**Step 1**: Open terminal and navigate to project directory:
```bash
cd "N:\CAI\sem 6\BGA\prj"
```

**Step 2**: Launch Spark Shell:
```bash
spark-shell --master local[*] --driver-memory 4g
```

**Step 3**: Load and run the Scala script:
```scala
:load scripts/scala/DataAnalysis.scala

// Run data analysis
DataAnalysis.analyzeTaxiData()

// Train ML model
DataAnalysis.trainSimpleModel()
```

### Method 2: Using Batch Script (Windows)

Double-click `run_scala_spark.bat` or run:
```cmd
run_scala_spark.bat
```

## 📊 Available Scala Functions

### 1. **Data Analysis**
```scala
DataAnalysis.analyzeTaxiData()
```

**What it does:**
- Loads 46M cleaned taxi records
- Shows schema and statistics
- Analyzes trip distribution by hour
- Compares rush hour vs non-rush hour
- Compares weekend vs weekday patterns

**Output:**
- Statistical summaries
- Aggregated metrics by time periods
- Average fares and distances

### 2. **Model Training**
```scala
DataAnalysis.trainSimpleModel()
```

**What it does:**
- Samples 10% of data (4.6M records)
- Creates feature vectors
- Normalizes features with StandardScaler
- Trains Random Forest Regressor (50 trees)
- Evaluates with RMSE and R²
- Shows feature importance

**Features used:**
- `trip_distance`
- `trip_duration_minutes`
- `passenger_count`
- `hour_of_day`
- `day_of_week`

**Output:**
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Feature importance ranking
- Sample predictions

## 🔧 Advanced Usage

### Custom Data Path
```scala
DataAnalysis.analyzeTaxiData("custom/path/to/data")
DataAnalysis.trainSimpleModel("custom/path/to/data")
```

### Modify Configuration
```scala
val spark = SparkSession.builder()
  .appName("CustomApp")
  .master("local[*]")
  .config("spark.driver.memory", "8g")
  .config("spark.executor.memory", "4g")
  .config("spark.sql.shuffle.partitions", "100")
  .getOrCreate()
```

### Save Model
```scala
// In spark-shell after training
model.write.overwrite().save("output/models/baseline_model")

// Load later
import org.apache.spark.ml.regression.RandomForestRegressionModel
val loadedModel = RandomForestRegressionModel.load("output/models/baseline_model")
```

## 📈 Performance Tips

### 1. Memory Configuration
```bash
# For large datasets
spark-shell --driver-memory 8g --executor-memory 8g
```

### 2. Partition Tuning
```scala
spark.conf.set("spark.sql.shuffle.partitions", "200")
df.repartition(100)
```

### 3. Caching Hot Data
```scala
val df = spark.read.parquet("data")
df.cache()  // Keep in memory
df.count()  // Trigger caching
```

### 4. Sample for Development
```scala
val sampleDf = df.sample(0.01)  // 1% sample for testing
```

## 🎓 Scala Spark Advantages

| Feature | PySpark | Scala Spark |
|---------|---------|-------------|
| **Performance** | Slow (serialization overhead) | Fast (native JVM) |
| **Type Safety** | Runtime errors | Compile-time checks |
| **New Features** | Delayed | Immediate access |
| **Community** | Large | Smaller but expert |
| **Learning Curve** | Easy | Moderate |

## 📝 Example: Full Analysis Workflow

```scala
// 1. Start Spark Shell
spark-shell --master local[*] --driver-memory 4g

// 2. Load script
:load scripts/scala/DataAnalysis.scala

// 3. Run analysis
DataAnalysis.analyzeTaxiData()

// 4. Train model
DataAnalysis.trainSimpleModel()

// 5. Custom analysis
import org.apache.spark.sql.functions._

val df = spark.read.parquet("output/output/processed/taxi_cleaned")

// Top 10 most expensive trips
df.orderBy(desc("fare_amount")).show(10, false)

// Average fare by day of week
df.groupBy("day_of_week")
  .agg(avg("fare_amount").alias("avg_fare"))
  .orderBy("day_of_week")
  .show()

// Trips above $100
df.filter($"fare_amount" > 100).count()

// 6. Exit
:quit
```

## 🐛 Troubleshooting

### Issue: `spark-shell` not found
**Solution:**
```cmd
set SPARK_HOME=C:\spark-3.5.1
set PATH=%PATH%;%SPARK_HOME%\bin
```

### Issue: OutOfMemoryError
**Solution:**
```bash
spark-shell --driver-memory 8g --executor-memory 8g
```

### Issue: Data not found
**Solution:** Check path:
```scala
import java.io.File
new File("output/output/processed/taxi_cleaned").exists()
```

### Issue: Scala version mismatch
**Solution:** Use Scala 2.12 (Spark 3.5.1 compatible):
```bash
spark-shell --version  # Check version
```

## 🔗 Next Steps

After running Scala analysis:

1. **Feature Engineering**: Run `FeatureEngineering.scala` (once enrichment completes)
2. **Bias Detection**: Train baseline vs fair models
3. **Visualization**: Export results for Python visualization
4. **Report**: Use findings in final report

## 📚 Resources

- [Spark Scala API Docs](https://spark.apache.org/docs/latest/api/scala/)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Scala Documentation](https://docs.scala-lang.org/)

---

**Happy Coding with Scala Spark! 🚀**
