// ============================================================
// NYC Taxi Fairness Audit  –  Stage 1: Data Cleaning
// ============================================================
// Big Data Frameworks: Apache Spark (Scala), Spark SQL, HDFS
//
// Run:
//   spark-shell --driver-memory 8g -i scripts/scala/01_DataCleaning.scala
//
// Input:  Local CSV  data/*.csv  OR  HDFS  /bigdata/taxi/raw/*.csv
// Output: Parquet    output/processed/taxi_cleaned
//         (also written to HDFS /bigdata/processed/taxi_cleaned when cluster mode)
//
// This script:
//   1. Loads 47M+ raw taxi records from CSV (with explicit schema)
//   2. Drops rows with NULL critical fields
//   3. Filters outliers (fare, distance, duration, speed, GPS bounding box)
//   4. Derives temporal features: hour, day-of-week, rush-hour, weekend, night
//   5. Computes trip_duration_minutes, fare_per_mile, speed_mph
//   6. Assigns a deterministic trip_id (sha2 hash of key fields)
//   7. Writes Snappy-compressed Parquet for downstream Spark & Hive
// ============================================================

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

println("\n" + "=" * 70)
println("  NYC TAXI FAIRNESS AUDIT – DATA CLEANING (Scala / Spark SQL)")
println("  Spark " + spark.version + "  |  Scala " + util.Properties.versionString)
println("=" * 70)

// ──────────────── Configuration ────────────────
val LOCAL_MODE     = true
val INPUT_PATH     = if (LOCAL_MODE) "data/yellow_tripdata_*.csv"
                     else "hdfs:///bigdata/taxi/raw/*.csv"
val OUTPUT_PATH    = if (LOCAL_MODE) "output/processed/taxi_cleaned"
                     else "hdfs:///bigdata/processed/taxi_cleaned"

// Data-quality thresholds
val MAX_FARE       = 500.0    // USD
val MIN_FARE       = 2.50     // NYC base fare
val MAX_DISTANCE   = 100.0    // miles
val MAX_DURATION   = 300.0    // minutes (5 hours)
val MAX_SPEED      = 100.0    // mph
val NYC_LAT_MIN    = 40.49;  val NYC_LAT_MAX = 40.92
val NYC_LON_MIN    = -74.27; val NYC_LON_MAX = -73.68

// ──────────────── Step 1: Load raw CSV with explicit schema ────────────────
println("\n[1/5] Loading raw taxi CSV data...")

val rawSchema = StructType(Array(
  StructField("VendorID",              IntegerType,   true),
  StructField("tpep_pickup_datetime",  TimestampType, true),
  StructField("tpep_dropoff_datetime", TimestampType, true),
  StructField("passenger_count",       IntegerType,   true),
  StructField("trip_distance",         DoubleType,    true),
  StructField("pickup_longitude",      DoubleType,    true),
  StructField("pickup_latitude",       DoubleType,    true),
  StructField("RatecodeID",            IntegerType,   true),
  StructField("store_and_fwd_flag",    StringType,    true),
  StructField("dropoff_longitude",     DoubleType,    true),
  StructField("dropoff_latitude",      DoubleType,    true),
  StructField("payment_type",          IntegerType,   true),
  StructField("fare_amount",           DoubleType,    true),
  StructField("extra",                 DoubleType,    true),
  StructField("mta_tax",               DoubleType,    true),
  StructField("tip_amount",            DoubleType,    true),
  StructField("tolls_amount",          DoubleType,    true),
  StructField("improvement_surcharge", DoubleType,    true),
  StructField("total_amount",          DoubleType,    true)
))

val rawDF = spark.read
  .option("header", "true")
  .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
  .schema(rawSchema)
  .csv(INPUT_PATH)

val initialCount = rawDF.count()
println(f"   Loaded $initialCount%,d raw records")

// ──────────────── Step 2: Remove NULL critical fields ────────────────
println("\n[2/5] Removing rows with NULL critical fields...")

val criticalCols = Seq("tpep_pickup_datetime", "tpep_dropoff_datetime",
  "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
  "trip_distance", "fare_amount")

var cleanDF = rawDF.na.drop(criticalCols)

// Remove zero GPS (invalid sensor readings)
cleanDF = cleanDF.filter(
  col("pickup_longitude")  =!= 0.0 && col("pickup_latitude")  =!= 0.0 &&
  col("dropoff_longitude") =!= 0.0 && col("dropoff_latitude") =!= 0.0
)

val afterNullCount = cleanDF.count()
println(f"   After NULL removal: $afterNullCount%,d  (dropped ${initialCount - afterNullCount}%,d)")

// ──────────────── Step 3: Derive computed columns ────────────────
println("\n[3/5] Computing derived fields...")

cleanDF = cleanDF
  // Trip duration in minutes
  .withColumn("trip_duration_minutes",
    round((unix_timestamp(col("tpep_dropoff_datetime")) -
           unix_timestamp(col("tpep_pickup_datetime"))) / 60.0, 2))
  // Average speed (mph)
  .withColumn("speed_mph",
    when(col("trip_duration_minutes") > 0,
      round(col("trip_distance") / (col("trip_duration_minutes") / 60.0), 2))
    .otherwise(0.0))
  // Temporal features
  .withColumn("hour_of_day", hour(col("tpep_pickup_datetime")))
  .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
  // Rush hour: 7-9 AM or 5-7 PM on weekdays (Mon=2..Fri=6 in Spark)
  .withColumn("is_rush_hour",
    ((col("hour_of_day").between(7, 9) || col("hour_of_day").between(17, 19)) &&
      col("day_of_week").between(2, 6)))
  // Weekend: Saturday=7, Sunday=1 in Spark
  .withColumn("is_weekend", col("day_of_week").isin(1, 7))
  // Night: 10 PM - 6 AM
  .withColumn("is_night", col("hour_of_day") >= 22 || col("hour_of_day") <= 5)
  // Fare per mile
  .withColumn("fare_per_mile",
    round(col("fare_amount") / col("trip_distance"), 2))
  // Grid-based pickup zone (0.01° ≈ 1.1 km)
  .withColumn("pickup_zone",
    concat(bround(col("pickup_latitude"), 2).cast("string"), lit("_"),
           bround(col("pickup_longitude"), 2).cast("string")))
  // Deterministic trip_id = SHA-256 of key fields → first 16 hex chars
  .withColumn("trip_id",
    substring(sha2(concat_ws("|",
      col("tpep_pickup_datetime").cast("string"),
      col("pickup_latitude").cast("string"),
      col("pickup_longitude").cast("string"),
      col("fare_amount").cast("string"),
      col("trip_distance").cast("string")), 256), 1, 16))

println("   Added: trip_id, trip_duration_minutes, speed_mph, hour_of_day, " +
        "day_of_week, is_rush_hour, is_weekend, is_night, fare_per_mile, pickup_zone")

// ──────────────── Step 4: Filter outliers ────────────────
println("\n[4/5] Filtering outliers...")

cleanDF = cleanDF.filter(
  // Fare range
  col("fare_amount").between(MIN_FARE, MAX_FARE) &&
  // Distance range
  col("trip_distance") > 0 && col("trip_distance") <= MAX_DISTANCE &&
  // Duration range
  col("trip_duration_minutes") > 0 && col("trip_duration_minutes") <= MAX_DURATION &&
  // Speed sanity
  col("speed_mph") <= MAX_SPEED &&
  // Passengers
  col("passenger_count") > 0 && col("passenger_count") <= 9 &&
  // NYC bounding box
  col("pickup_latitude").between(NYC_LAT_MIN, NYC_LAT_MAX) &&
  col("pickup_longitude").between(NYC_LON_MIN, NYC_LON_MAX) &&
  col("dropoff_latitude").between(NYC_LAT_MIN, NYC_LAT_MAX) &&
  col("dropoff_longitude").between(NYC_LON_MIN, NYC_LON_MAX)
)

val finalCount = cleanDF.count()
val removedPct = ((initialCount - finalCount).toDouble / initialCount * 100)
println(f"   After outlier removal: $finalCount%,d")
println(f"   Total removed: ${initialCount - finalCount}%,d ($removedPct%.1f%%)")

// ──────────────── Step 5: Select final columns & write Parquet ────────────────
println(s"\n[5/5] Writing Parquet to: $OUTPUT_PATH")

val finalColumns = Seq(
  "trip_id",
  "tpep_pickup_datetime", "tpep_dropoff_datetime",
  "passenger_count", "trip_distance",
  "pickup_longitude", "pickup_latitude",
  "dropoff_longitude", "dropoff_latitude",
  "fare_amount", "total_amount",
  "trip_duration_minutes", "speed_mph",
  "hour_of_day", "day_of_week",
  "is_rush_hour", "is_weekend", "is_night",
  "fare_per_mile", "pickup_zone"
)

val outputDF = cleanDF.select(finalColumns.map(col): _*)

outputDF.write
  .mode("overwrite")
  .option("compression", "snappy")
  .parquet(OUTPUT_PATH)

// ──────────────── Summary ────────────────
println("\n" + "=" * 70)
println("  DATA CLEANING COMPLETE")
println("=" * 70)
println(f"  Initial records:  $initialCount%,d")
println(f"  Final records:    $finalCount%,d")
println(f"  Records removed:  ${initialCount - finalCount}%,d ($removedPct%.1f%%)")
println(f"  Output:           $OUTPUT_PATH")
println()

println("  Data Quality Filters Applied:")
println(f"    Fare range:     $$${MIN_FARE}%.2f – $$${MAX_FARE}%.0f")
println(f"    Distance range: 0 – $MAX_DISTANCE%.0f miles")
println(f"    Duration range: 0 – $MAX_DURATION%.0f minutes")
println(f"    Max speed:      $MAX_SPEED%.0f mph")
println(f"    GPS bounding:   NYC metro area")
println()

println("  Schema:")
outputDF.printSchema()

println("  Sample data:")
outputDF.show(5, truncate = false)

println("  Fare distribution:")
outputDF.describe("fare_amount", "trip_distance", "trip_duration_minutes", "speed_mph")
  .show()

println("=" * 70)
println("  Next: Run 02_DataEnrichment.scala")
println("=" * 70)

System.exit(0)
