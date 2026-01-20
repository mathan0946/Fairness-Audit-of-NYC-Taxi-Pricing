"""
============================================
NYC Taxi Fairness Audit - Data Cleaning
============================================
Stage 3A: Clean raw taxi data using PySpark

This script:
1. Loads raw taxi data from HDFS/local
2. Removes missing values
3. Removes outliers
4. Calculates derived fields
5. Saves cleaned data as Parquet

Author: Big Data Analytics Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, hour, dayofweek, unix_timestamp,
    lit, monotonically_increasing_id, concat, 
    round as spark_round, abs as spark_abs
)
from pyspark.sql.types import *
import os
import sys

# ============================================
# CONFIGURATION
# ============================================

# For local testing (without HDFS)
LOCAL_MODE = True

if LOCAL_MODE:
    INPUT_PATH = "data/yellow_tripdata_*.csv"
    OUTPUT_PATH = "output/processed/taxi_cleaned"
    CENSUS_PATH = "data/us_income_zipcode.csv"
else:
    INPUT_PATH = "hdfs:///bigdata/taxi/raw/*.csv"
    OUTPUT_PATH = "hdfs:///bigdata/processed/taxi_cleaned"
    CENSUS_PATH = "hdfs:///bigdata/census/us_income_zipcode.csv"

# Data quality thresholds
MAX_FARE = 500  # Maximum reasonable fare
MAX_DISTANCE = 100  # Maximum reasonable distance (miles)
MAX_DURATION = 300  # Maximum trip duration (minutes)
MAX_SPEED = 100  # Maximum reasonable speed (mph)
MIN_FARE = 2.5  # Minimum fare (base fare)

# ============================================
# SPARK SESSION
# ============================================

def create_spark_session():
    """Create and configure Spark session."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Fairness_DataCleaning") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================
# DATA LOADING
# ============================================

def load_taxi_data(spark, input_path):
    """Load raw taxi data from CSV files."""
    print(f"\n[1/5] Loading taxi data from: {input_path}")
    
    schema = StructType([
        StructField("VendorID", IntegerType(), True),
        StructField("tpep_pickup_datetime", TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("RatecodeID", IntegerType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("payment_type", IntegerType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("total_amount", DoubleType(), True)
    ])
    
    df = spark.read.csv(
        input_path,
        header=True,
        schema=schema,
        timestampFormat="yyyy-MM-dd HH:mm:ss"
    )
    
    initial_count = df.count()
    print(f"   Loaded {initial_count:,} records")
    
    return df, initial_count


# ============================================
# DATA CLEANING FUNCTIONS
# ============================================

def remove_missing_values(df):
    """Remove rows with critical missing values."""
    print("\n[2/5] Removing missing values...")
    
    # Define critical columns that cannot be null
    critical_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "trip_distance",
        "fare_amount"
    ]
    
    # Remove nulls
    df_clean = df.dropna(subset=critical_columns)
    
    # Remove zero coordinates (invalid GPS)
    df_clean = df_clean.filter(
        (col("pickup_longitude") != 0) & 
        (col("pickup_latitude") != 0) &
        (col("dropoff_longitude") != 0) & 
        (col("dropoff_latitude") != 0)
    )
    
    cleaned_count = df_clean.count()
    print(f"   Records after removing nulls: {cleaned_count:,}")
    
    return df_clean


def remove_outliers(df):
    """Remove statistical outliers and invalid data."""
    print("\n[3/5] Removing outliers...")
    
    # Calculate trip duration in minutes
    df = df.withColumn(
        "trip_duration_minutes",
        (unix_timestamp("tpep_dropoff_datetime") - 
         unix_timestamp("tpep_pickup_datetime")) / 60
    )
    
    # Calculate speed (mph)
    df = df.withColumn(
        "speed_mph",
        when(col("trip_duration_minutes") > 0,
             col("trip_distance") / (col("trip_duration_minutes") / 60)
        ).otherwise(0)
    )
    
    # Apply filters for valid trips
    df_clean = df.filter(
        # Valid fare range
        (col("fare_amount") >= MIN_FARE) &
        (col("fare_amount") <= MAX_FARE) &
        # Valid distance
        (col("trip_distance") > 0) &
        (col("trip_distance") <= MAX_DISTANCE) &
        # Valid duration
        (col("trip_duration_minutes") > 0) &
        (col("trip_duration_minutes") <= MAX_DURATION) &
        # Valid speed (not teleporting!)
        (col("speed_mph") <= MAX_SPEED) &
        # Valid passenger count
        (col("passenger_count") > 0) &
        (col("passenger_count") <= 9) &
        # NYC bounding box (roughly)
        (col("pickup_longitude").between(-74.5, -73.5)) &
        (col("pickup_latitude").between(40.4, 41.0)) &
        (col("dropoff_longitude").between(-74.5, -73.5)) &
        (col("dropoff_latitude").between(40.4, 41.0))
    )
    
    # Drop speed column (only needed for filtering)
    df_clean = df_clean.drop("speed_mph")
    
    cleaned_count = df_clean.count()
    print(f"   Records after removing outliers: {cleaned_count:,}")
    
    return df_clean


def add_derived_fields(df):
    """Calculate derived fields for analysis."""
    print("\n[4/5] Adding derived fields...")
    
    # Add unique trip ID
    df = df.withColumn(
        "trip_id",
        concat(
            col("VendorID").cast("string"),
            lit("_"),
            monotonically_increasing_id().cast("string")
        )
    )
    
    # Extract time-based features
    df = df.withColumn("hour_of_day", hour("tpep_pickup_datetime"))
    df = df.withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))
    
    # Rush hour indicator (7-9 AM or 5-7 PM on weekdays)
    df = df.withColumn(
        "is_rush_hour",
        when(
            ((col("hour_of_day").between(7, 9)) | 
             (col("hour_of_day").between(17, 19))) &
            (col("day_of_week").between(2, 6)),  # Monday=2 to Friday=6 in Spark
            True
        ).otherwise(False)
    )
    
    # Weekend indicator
    df = df.withColumn(
        "is_weekend",
        when(
            (col("day_of_week") == 1) | (col("day_of_week") == 7),
            True
        ).otherwise(False)
    )
    
    # Night trip indicator (10 PM - 6 AM)
    df = df.withColumn(
        "is_night",
        when(
            (col("hour_of_day") >= 22) | (col("hour_of_day") <= 6),
            True
        ).otherwise(False)
    )
    
    # Fare per mile (price efficiency)
    df = df.withColumn(
        "fare_per_mile",
        spark_round(col("fare_amount") / col("trip_distance"), 2)
    )
    
    # Round trip duration
    df = df.withColumn(
        "trip_duration_minutes",
        spark_round(col("trip_duration_minutes"), 2)
    )
    
    print("   Added: trip_id, hour_of_day, day_of_week, is_rush_hour, is_weekend, is_night, fare_per_mile")
    
    return df


def assign_zip_codes(df):
    """
    Assign ZIP codes based on pickup coordinates.
    In production, use reverse geocoding or spatial join with NYC ZIP boundaries.
    This is a simplified approximation using coordinate binning.
    """
    print("   Assigning approximate ZIP codes from coordinates...")
    
    # Create approximate ZIP code zones based on latitude/longitude
    # This is a simplified approach - in production, use proper geocoding
    df = df.withColumn(
        "pickup_zone",
        concat(
            spark_round(col("pickup_latitude"), 2).cast("string"),
            lit("_"),
            spark_round(col("pickup_longitude"), 2).cast("string")
        )
    )
    
    return df


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC TAXI FAIRNESS AUDIT - DATA CLEANING")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Load data
        df, initial_count = load_taxi_data(spark, INPUT_PATH)
        
        # Clean data
        df = remove_missing_values(df)
        df = remove_outliers(df)
        df = add_derived_fields(df)
        df = assign_zip_codes(df)
        
        # Select final columns
        final_columns = [
            "trip_id",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "fare_amount",
            "total_amount",
            "trip_duration_minutes",
            "hour_of_day",
            "day_of_week",
            "is_rush_hour",
            "is_weekend",
            "is_night",
            "fare_per_mile",
            "pickup_zone"
        ]
        
        df_final = df.select(final_columns)
        
        # Save cleaned data
        print(f"\n[5/5] Saving cleaned data to: {OUTPUT_PATH}")
        
        df_final.write \
            .mode("overwrite") \
            .parquet(OUTPUT_PATH)
        
        # Print summary
        final_count = df_final.count()
        removed_pct = ((initial_count - final_count) / initial_count) * 100
        
        print("\n" + "=" * 60)
        print("DATA CLEANING COMPLETE")
        print("=" * 60)
        print(f"Initial records:  {initial_count:,}")
        print(f"Final records:    {final_count:,}")
        print(f"Records removed:  {initial_count - final_count:,} ({removed_pct:.1f}%)")
        print(f"Output location:  {OUTPUT_PATH}")
        print("=" * 60)
        
        # Show sample
        print("\nSample of cleaned data:")
        df_final.show(5, truncate=False)
        
        # Print schema
        print("\nFinal schema:")
        df_final.printSchema()
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
