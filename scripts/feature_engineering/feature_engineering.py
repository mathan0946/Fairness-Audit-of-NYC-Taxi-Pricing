"""
============================================
NYC Taxi Fairness Audit - Feature Engineering (Python)
============================================
Stage 3C: Create ML-ready features using PySpark

Alternative Python implementation of feature engineering.
Use this if you prefer Python over Scala.

Author: Big Data Analytics Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, round as spark_round
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
)
from pyspark.ml import Pipeline

# ============================================
# CONFIGURATION
# ============================================

LOCAL_MODE = True

if LOCAL_MODE:
    INPUT_PATH = "output/processed/taxi_enriched"
    OUTPUT_PATH = "output/processed/ml_ready"
else:
    INPUT_PATH = "hdfs:///bigdata/processed/taxi_enriched"
    OUTPUT_PATH = "hdfs:///bigdata/processed/ml_ready"


def create_spark_session():
    """Create Spark session."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Fairness_FeatureEngineering_Python") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    print("=" * 60)
    print("NYC TAXI FAIRNESS AUDIT - FEATURE ENGINEERING (PYTHON)")
    print("=" * 60)
    
    spark = create_spark_session()
    
    try:
        # Step 1: Load data
        print(f"\n[1/5] Loading enriched data from: {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH)
        print(f"   Loaded {df.count():,} records")
        
        # Step 2: Select features
        print("\n[2/5] Preparing feature columns...")
        
        feature_columns = [
            "trip_id", "trip_distance", "trip_duration_minutes",
            "passenger_count", "hour_of_day", "day_of_week",
            "is_rush_hour", "is_weekend", "is_night",
            "borough", "income_category", "median_income",
            "fare_amount"
        ]
        
        df_features = df.select(feature_columns)
        df_features = df_features.na.drop()
        
        # Convert booleans to doubles
        df_features = df_features \
            .withColumn("is_rush_hour_num", 
                       when(col("is_rush_hour"), 1.0).otherwise(0.0)) \
            .withColumn("is_weekend_num", 
                       when(col("is_weekend"), 1.0).otherwise(0.0)) \
            .withColumn("is_night_num", 
                       when(col("is_night"), 1.0).otherwise(0.0))
        
        print(f"   Records after preparation: {df_features.count():,}")
        
        # Step 3: Encode categorical variables
        print("\n[3/5] Encoding categorical variables...")
        
        # String indexers
        borough_indexer = StringIndexer(
            inputCol="borough",
            outputCol="borough_index",
            handleInvalid="keep"
        )
        
        income_indexer = StringIndexer(
            inputCol="income_category",
            outputCol="income_category_index",
            handleInvalid="keep"
        )
        
        # One-hot encoders
        borough_encoder = OneHotEncoder(
            inputCol="borough_index",
            outputCol="borough_encoded"
        )
        
        income_encoder = OneHotEncoder(
            inputCol="income_category_index",
            outputCol="income_category_encoded"
        )
        
        # Step 4: Create feature vectors
        print("\n[4/5] Creating feature vectors...")
        
        # BASELINE features (includes location/income - potentially biased)
        baseline_feature_cols = [
            "trip_distance", "trip_duration_minutes", "passenger_count",
            "hour_of_day", "day_of_week",
            "is_rush_hour_num", "is_weekend_num", "is_night_num",
            "median_income"  # This could introduce bias!
        ]
        
        baseline_assembler = VectorAssembler(
            inputCols=baseline_feature_cols,
            outputCol="features_baseline_raw",
            handleInvalid="skip"
        )
        
        # FAIR features (excludes location/income)
        fair_feature_cols = [
            "trip_distance", "trip_duration_minutes", "passenger_count",
            "hour_of_day", "day_of_week",
            "is_rush_hour_num", "is_weekend_num", "is_night_num"
            # NO income or location features!
        ]
        
        fair_assembler = VectorAssembler(
            inputCols=fair_feature_cols,
            outputCol="features_fair_raw",
            handleInvalid="skip"
        )
        
        # Scalers
        baseline_scaler = StandardScaler(
            inputCol="features_baseline_raw",
            outputCol="features_baseline",
            withMean=True,
            withStd=True
        )
        
        fair_scaler = StandardScaler(
            inputCol="features_fair_raw",
            outputCol="features_fair",
            withMean=True,
            withStd=True
        )
        
        # Build pipeline
        pipeline = Pipeline(stages=[
            borough_indexer,
            borough_encoder,
            income_indexer,
            income_encoder,
            baseline_assembler,
            fair_assembler,
            baseline_scaler,
            fair_scaler
        ])
        
        # Fit and transform
        model = pipeline.fit(df_features)
        df_transformed = model.transform(df_features)
        
        # Add label column for MLlib
        df_transformed = df_transformed.withColumn("label", col("fare_amount"))
        
        # Step 5: Save
        print(f"\n[5/5] Saving ML-ready data to: {OUTPUT_PATH}")
        
        final_columns = [
            "trip_id", "features_baseline", "features_fair", "label",
            "fare_amount", "income_category", "income_category_index",
            "borough", "borough_index", "trip_distance", "median_income"
        ]
        
        df_final = df_transformed.select(final_columns)
        df_final.write.mode("overwrite").parquet(OUTPUT_PATH)
        
        final_count = df_final.count()
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"ML-ready records: {final_count:,}")
        print(f"Output location:  {OUTPUT_PATH}")
        print("=" * 60)
        
        # Summary
        print("\nBaseline features (potentially biased):")
        for c in baseline_feature_cols:
            print(f"  - {c}")
        
        print("\nFair features (no location/income):")
        for c in fair_feature_cols:
            print(f"  - {c}")
        
        print("\nIncome category distribution:")
        df_final.groupBy("income_category").agg(
            count("*").alias("count"),
            spark_round(avg("fare_amount"), 2).alias("avg_fare")
        ).orderBy("income_category").show()
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
