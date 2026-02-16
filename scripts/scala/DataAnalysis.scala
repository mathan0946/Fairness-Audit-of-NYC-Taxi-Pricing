/**
 * ============================================
 * NYC Taxi Fairness Audit - Scala Data Analysis
 * ============================================
 * Interactive Scala Spark analysis of taxi data
 *
 * Usage in spark-shell:
 *   :load scripts/scala/DataAnalysis.scala
 *   DataAnalysis.analyzeTaxiData()
 *
 * Author: Big Data Analytics Project
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

object DataAnalysis {

  def analyzeTaxiData(inputPath: String = "output/output/processed/taxi_cleaned"): Unit = {
    println("=" * 70)
    println("NYC TAXI FAIRNESS AUDIT - SCALA SPARK ANALYSIS")
    println("=" * 70)
    
    val spark = SparkSession.builder()
      .appName("TaxiFairnessAnalysis")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()
    
    import spark.implicits._
    
    // Load cleaned data
    println(s"\n[1/6] Loading data from: $inputPath")
    val df = spark.read.parquet(inputPath)
    val recordCount = df.count()
    println(s"✓ Loaded $recordCount records")
    
    // Data overview
    println("\n[2/6] Data Schema:")
    df.printSchema()
    
    // Statistical summary
    println("\n[3/6] Statistical Summary:")
    df.select(
      "fare_amount", 
      "trip_distance", 
      "trip_duration_minutes",
      "fare_per_mile"
    ).describe().show()
    
    // Distribution by hour
    println("\n[4/6] Trip Distribution by Hour of Day:")
    df.groupBy("hour_of_day")
      .agg(
        count("*").alias("trip_count"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_distance").alias("avg_distance")
      )
      .orderBy("hour_of_day")
      .show(24, false)
    
    // Rush hour analysis
    println("\n[5/6] Rush Hour vs Non-Rush Hour Comparison:")
    df.groupBy("is_rush_hour")
      .agg(
        count("*").alias("trip_count"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration_minutes").alias("avg_duration"),
        avg("fare_per_mile").alias("avg_fare_per_mile")
      )
      .show(false)
    
    // Weekend vs Weekday
    println("\n[6/6] Weekend vs Weekday Comparison:")
    df.groupBy("is_weekend")
      .agg(
        count("*").alias("trip_count"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_distance").alias("avg_distance"),
        avg("fare_per_mile").alias("avg_fare_per_mile")
      )
      .show(false)
    
    println("\n" + "=" * 70)
    println("Analysis complete!")
    println("=" * 70)
  }
  
  def trainSimpleModel(inputPath: String = "output/output/processed/taxi_cleaned"): Unit = {
    println("=" * 70)
    println("TRAINING BASELINE MODEL WITH SCALA")
    println("=" * 70)
    
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    // Load and prepare data
    println("\n[1/4] Loading and preparing data...")
    val df = spark.read.parquet(inputPath)
      .na.drop()
      .sample(0.1)  // Sample 10% for faster training
    
    println(s"✓ Loaded ${df.count()} sampled records")
    
    // Feature engineering
    println("\n[2/4] Creating feature vectors...")
    val featureCols = Array(
      "trip_distance",
      "trip_duration_minutes",
      "passenger_count",
      "hour_of_day",
      "day_of_week"
    )
    
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")
    
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)
    
    val assembled = assembler.transform(df)
    val scalerModel = scaler.fit(assembled)
    val scaled = scalerModel.transform(assembled)
    
    // Split data
    val Array(trainData, testData) = scaled.randomSplit(Array(0.7, 0.3), seed = 42)
    
    println(s"✓ Training set: ${trainData.count()} records")
    println(s"✓ Test set: ${testData.count()} records")
    
    // Train model
    println("\n[3/4] Training Random Forest model...")
    val rf = new RandomForestRegressor()
      .setLabelCol("fare_amount")
      .setFeaturesCol("features")
      .setNumTrees(50)
      .setMaxDepth(10)
      .setSeed(42)
    
    val model = rf.fit(trainData)
    println("✓ Model trained!")
    
    // Evaluate
    println("\n[4/4] Evaluating model...")
    val predictions = model.transform(testData)
    
    val evaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("fare_amount")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    
    val evaluatorR2 = new RegressionEvaluator()
      .setLabelCol("fare_amount")
      .setPredictionCol("prediction")
      .setMetricName("r2")
    
    val rmse = evaluatorRMSE.evaluate(predictions)
    val r2 = evaluatorR2.evaluate(predictions)
    
    println(s"\n✓ Model Performance:")
    println(s"   RMSE: ${"%.2f".format(rmse)}")
    println(s"   R²: ${"%.4f".format(r2)}")
    
    // Show feature importance
    println("\n✓ Feature Importance:")
    model.featureImportances.toArray.zip(featureCols).sortBy(-_._1).foreach {
      case (importance, feature) => 
        println(f"   $feature%-25s: ${importance * 100}%.2f%%")
    }
    
    // Sample predictions
    println("\n✓ Sample Predictions:")
    predictions.select("fare_amount", "prediction", "trip_distance", "trip_duration_minutes")
      .limit(10)
      .show(false)
    
    println("\n" + "=" * 70)
    println("Model training complete!")
    println("=" * 70)
  }
}

// Make it easy to run
println("""
╔═══════════════════════════════════════════════════════════════════════╗
║                 NYC Taxi Fairness Audit - Scala Loaded                ║
╚═══════════════════════════════════════════════════════════════════════╝

Available methods:
  DataAnalysis.analyzeTaxiData()        - Run data analysis
  DataAnalysis.trainSimpleModel()       - Train ML model

Example usage:
  DataAnalysis.analyzeTaxiData("output/output/processed/taxi_cleaned")
""")
