"""
============================================
NYC Taxi Fairness Audit - Baseline Model
============================================
Stage 4A: Train baseline ML model (with location/income features)

This model INCLUDES location and income features, which may
introduce algorithmic bias against low-income neighborhoods.

Author: Big Data Analytics Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round as spark_round, abs as spark_abs
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# ============================================
# CONFIGURATION
# ============================================

LOCAL_MODE = True

if LOCAL_MODE:
    INPUT_PATH = "output/processed/ml_ready"
    MODEL_PATH = "output/models/baseline_model"
    PREDICTIONS_PATH = "output/results/baseline_predictions"
else:
    INPUT_PATH = "hdfs:///bigdata/processed/ml_ready"
    MODEL_PATH = "hdfs:///bigdata/models/baseline_model"
    PREDICTIONS_PATH = "hdfs:///bigdata/results/baseline_predictions"

# Model hyperparameters
NUM_TREES = 100
MAX_DEPTH = 10
TRAIN_RATIO = 0.8
SEED = 42


def create_spark_session():
    """Create Spark session."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Fairness_BaselineModel") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, input_path):
    """Load ML-ready data."""
    print(f"\n[1/5] Loading ML-ready data from: {input_path}")
    
    df = spark.read.parquet(input_path)
    record_count = df.count()
    print(f"   Loaded {record_count:,} records")
    
    return df


def split_data(df, train_ratio=0.8, seed=42):
    """Split data into training and test sets."""
    print(f"\n[2/5] Splitting data ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test)...")
    
    train_df, test_df = df.randomSplit([train_ratio, 1-train_ratio], seed=seed)
    
    print(f"   Training records: {train_df.count():,}")
    print(f"   Test records:     {test_df.count():,}")
    
    return train_df, test_df


def train_baseline_model(train_df):
    """Train Random Forest model with baseline features (includes income)."""
    print("\n[3/5] Training baseline Random Forest model...")
    print(f"   Number of trees: {NUM_TREES}")
    print(f"   Max depth: {MAX_DEPTH}")
    print("   Features: BASELINE (includes income/location)")
    
    # Random Forest Regressor
    rf = RandomForestRegressor(
        featuresCol="features_baseline",
        labelCol="label",
        predictionCol="predicted_fare_baseline",
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        seed=SEED
    )
    
    # Train model
    model = rf.fit(train_df)
    
    print("   Model training complete!")
    
    # Feature importances
    print("\n   Feature Importances:")
    feature_names = [
        "trip_distance", "trip_duration_minutes", "passenger_count",
        "hour_of_day", "day_of_week",
        "is_rush_hour", "is_weekend", "is_night",
        "median_income"
    ]
    importances = model.featureImportances.toArray()
    for name, importance in sorted(zip(feature_names, importances), 
                                    key=lambda x: x[1], reverse=True):
        print(f"     {name}: {importance:.4f}")
    
    return model


def evaluate_model(model, test_df, prediction_col="predicted_fare_baseline"):
    """Evaluate model performance."""
    print("\n[4/5] Evaluating model performance...")
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Calculate metrics
    evaluator_rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol=prediction_col,
        metricName="rmse"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="label",
        predictionCol=prediction_col,
        metricName="r2"
    )
    
    evaluator_mae = RegressionEvaluator(
        labelCol="label",
        predictionCol=prediction_col,
        metricName="mae"
    )
    
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    print(f"   RMSE:  ${rmse:.2f}")
    print(f"   MAE:   ${mae:.2f}")
    print(f"   R²:    {r2:.4f} ({r2*100:.1f}%)")
    
    # Add error column
    predictions = predictions.withColumn(
        "prediction_error",
        col(prediction_col) - col("label")
    )
    
    predictions = predictions.withColumn(
        "abs_error",
        spark_abs(col("prediction_error"))
    )
    
    return predictions, {"rmse": rmse, "r2": r2, "mae": mae}


def analyze_predictions_by_income(predictions):
    """Analyze predictions grouped by income category."""
    print("\n   Predictions by Income Category:")
    
    summary = predictions.groupBy("income_category").agg(
        count("*").alias("trip_count"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(avg("label"), 2).alias("avg_actual_fare"),
        spark_round(avg("predicted_fare_baseline"), 2).alias("avg_predicted_fare"),
        spark_round(avg("prediction_error"), 2).alias("avg_error"),
        spark_round(avg("abs_error"), 2).alias("avg_abs_error")
    ).orderBy("income_category")
    
    summary.show()
    
    # Calculate overcharge percentage
    print("   Overcharge Analysis:")
    summary_with_overcharge = predictions.groupBy("income_category").agg(
        count("*").alias("trips"),
        spark_round(avg("label"), 2).alias("actual"),
        spark_round(avg("predicted_fare_baseline"), 2).alias("predicted"),
        spark_round(
            (avg("predicted_fare_baseline") - avg("label")) / avg("label") * 100,
            1
        ).alias("overcharge_pct")
    ).orderBy("income_category")
    
    summary_with_overcharge.show()
    
    return summary


def save_results(model, predictions, model_path, predictions_path):
    """Save model and predictions."""
    print(f"\n[5/5] Saving results...")
    
    # Save model
    print(f"   Saving model to: {model_path}")
    model.write().overwrite().save(model_path)
    
    # Save predictions
    print(f"   Saving predictions to: {predictions_path}")
    
    # Select relevant columns
    output_columns = [
        "trip_id",
        "label",
        "predicted_fare_baseline",
        "prediction_error",
        "abs_error",
        "income_category",
        "borough",
        "trip_distance",
        "median_income"
    ]
    
    predictions.select(output_columns).write \
        .mode("overwrite") \
        .parquet(predictions_path)
    
    print("   Results saved successfully!")


def main():
    print("=" * 60)
    print("NYC TAXI FAIRNESS AUDIT - BASELINE MODEL TRAINING")
    print("=" * 60)
    print("\n⚠️  WARNING: This model includes income/location features")
    print("    which may introduce algorithmic bias!")
    
    spark = create_spark_session()
    
    try:
        # Load data
        df = load_data(spark, INPUT_PATH)
        
        # Split data
        train_df, test_df = split_data(df, TRAIN_RATIO, SEED)
        
        # Train model
        model = train_baseline_model(train_df)
        
        # Evaluate
        predictions, metrics = evaluate_model(model, test_df)
        
        # Analyze by income
        analyze_predictions_by_income(predictions)
        
        # Save results
        save_results(model, predictions, MODEL_PATH, PREDICTIONS_PATH)
        
        # Print summary
        print("\n" + "=" * 60)
        print("BASELINE MODEL TRAINING COMPLETE")
        print("=" * 60)
        print(f"Model Accuracy (R²): {metrics['r2']*100:.1f}%")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print("\nNext step: Run 02_fair_model.py to train fairness-aware model")
        print("=" * 60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
