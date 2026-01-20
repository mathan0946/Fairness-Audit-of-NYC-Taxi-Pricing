/**
 * ============================================
 * NYC Taxi Fairness Audit - Feature Engineering
 * ============================================
 * Stage 3C: Create ML-ready features using Scala Spark
 *
 * This script:
 * 1. Loads enriched taxi data
 * 2. Creates feature vectors for ML training
 * 3. Encodes categorical variables
 * 4. Normalizes numerical features
 * 5. Saves ML-ready data
 *
 * Author: Big Data Analytics Project
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler}
import org.apache.spark.ml.Pipeline

object FeatureEngineering {

  // Configuration
  val LOCAL_MODE = true
  
  val INPUT_PATH = if (LOCAL_MODE) 
    "output/processed/taxi_enriched" 
  else 
    "hdfs:///bigdata/processed/taxi_enriched"
  
  val OUTPUT_PATH = if (LOCAL_MODE) 
    "output/processed/ml_ready" 
  else 
    "hdfs:///bigdata/processed/ml_ready"

  def main(args: Array[String]): Unit = {
    println("=" * 60)
    println("NYC TAXI FAIRNESS AUDIT - FEATURE ENGINEERING (SCALA)")
    println("=" * 60)

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("NYC_Taxi_Fairness_FeatureEngineering")
      .config("spark.sql.parquet.compression.codec", "snappy")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    try {
      // ========================================
      // Step 1: Load enriched data
      // ========================================
      println(s"\n[1/5] Loading enriched data from: $INPUT_PATH")
      
      val df = spark.read.parquet(INPUT_PATH)
      println(s"   Loaded ${df.count()} records")
      
      // ========================================
      // Step 2: Select and prepare features
      // ========================================
      println("\n[2/5] Preparing feature columns...")
      
      // Select relevant columns for ML
      val featureColumns = Seq(
        "trip_id",
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "hour_of_day",
        "day_of_week",
        "is_rush_hour",
        "is_weekend",
        "is_night",
        "borough",
        "income_category",
        "median_income",
        "fare_amount"  // Target variable
      )
      
      var dfFeatures = df.select(featureColumns.map(col): _*)
      
      // Handle any remaining nulls
      dfFeatures = dfFeatures.na.drop()
      
      println(s"   Records after null removal: ${dfFeatures.count()}")
      
      // ========================================
      // Step 3: Encode categorical variables
      // ========================================
      println("\n[3/5] Encoding categorical variables...")
      
      // Borough encoding
      val boroughIndexer = new StringIndexer()
        .setInputCol("borough")
        .setOutputCol("borough_index")
        .setHandleInvalid("keep")
      
      val boroughEncoder = new OneHotEncoder()
        .setInputCol("borough_index")
        .setOutputCol("borough_encoded")
      
      // Income category encoding
      val incomeIndexer = new StringIndexer()
        .setInputCol("income_category")
        .setOutputCol("income_category_index")
        .setHandleInvalid("keep")
      
      val incomeEncoder = new OneHotEncoder()
        .setInputCol("income_category_index")
        .setOutputCol("income_category_encoded")
      
      // Convert boolean to double
      dfFeatures = dfFeatures
        .withColumn("is_rush_hour_num", when(col("is_rush_hour"), 1.0).otherwise(0.0))
        .withColumn("is_weekend_num", when(col("is_weekend"), 1.0).otherwise(0.0))
        .withColumn("is_night_num", when(col("is_night"), 1.0).otherwise(0.0))
      
      // ========================================
      // Step 4: Create feature vectors
      // ========================================
      println("\n[4/5] Creating feature vectors...")
      
      // Numerical features for the BASELINE model (includes location/income)
      val baselineFeatureColumns = Array(
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "hour_of_day",
        "day_of_week",
        "is_rush_hour_num",
        "is_weekend_num",
        "is_night_num",
        "median_income"  // Includes income - potentially biased
      )
      
      val baselineAssembler = new VectorAssembler()
        .setInputCols(baselineFeatureColumns)
        .setOutputCol("features_baseline_raw")
        .setHandleInvalid("skip")
      
      // Numerical features for the FAIR model (excludes location/income)
      val fairFeatureColumns = Array(
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "hour_of_day",
        "day_of_week",
        "is_rush_hour_num",
        "is_weekend_num",
        "is_night_num"
        // NO income or location features
      )
      
      val fairAssembler = new VectorAssembler()
        .setInputCols(fairFeatureColumns)
        .setOutputCol("features_fair_raw")
        .setHandleInvalid("skip")
      
      // Scalers for normalization
      val baselineScaler = new StandardScaler()
        .setInputCol("features_baseline_raw")
        .setOutputCol("features_baseline")
        .setWithMean(true)
        .setWithStd(true)
      
      val fairScaler = new StandardScaler()
        .setInputCol("features_fair_raw")
        .setOutputCol("features_fair")
        .setWithMean(true)
        .setWithStd(true)
      
      // Create pipeline
      val pipeline = new Pipeline().setStages(Array(
        boroughIndexer,
        boroughEncoder,
        incomeIndexer,
        incomeEncoder,
        baselineAssembler,
        fairAssembler,
        baselineScaler,
        fairScaler
      ))
      
      // Fit and transform
      val model = pipeline.fit(dfFeatures)
      var dfTransformed = model.transform(dfFeatures)
      
      // Rename fare_amount to label for MLlib
      dfTransformed = dfTransformed.withColumn("label", col("fare_amount"))
      
      // ========================================
      // Step 5: Save ML-ready data
      // ========================================
      println(s"\n[5/5] Saving ML-ready data to: $OUTPUT_PATH")
      
      // Select final columns
      val finalColumns = Seq(
        "trip_id",
        "features_baseline",
        "features_fair",
        "label",
        "fare_amount",
        "income_category",
        "income_category_index",
        "borough",
        "borough_index",
        "trip_distance",
        "median_income"
      )
      
      val dfFinal = dfTransformed.select(finalColumns.map(col): _*)
      
      dfFinal.write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
      
      val finalCount = dfFinal.count()
      
      println("\n" + "=" * 60)
      println("FEATURE ENGINEERING COMPLETE")
      println("=" * 60)
      println(s"ML-ready records: $finalCount")
      println(s"Output location:  $OUTPUT_PATH")
      println("=" * 60)
      
      // Show sample
      println("\nSample of ML-ready data:")
      dfFinal.show(5, truncate = false)
      
      // Show feature statistics
      println("\nBaseline feature columns (includes income):")
      baselineFeatureColumns.foreach(c => println(s"  - $c"))
      
      println("\nFair feature columns (excludes income/location):")
      fairFeatureColumns.foreach(c => println(s"  - $c"))
      
      // Distribution of income categories
      println("\nIncome category distribution:")
      dfFinal.groupBy("income_category")
        .agg(
          count("*").alias("count"),
          round(avg("fare_amount"), 2).alias("avg_fare")
        )
        .orderBy("income_category")
        .show()

    } finally {
      spark.stop()
    }
  }
}
