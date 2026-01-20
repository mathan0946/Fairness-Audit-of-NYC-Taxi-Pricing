-- ============================================
-- HIVE DDL SCRIPT: NYC Taxi Fairness Audit
-- ============================================

-- Create database
CREATE DATABASE IF NOT EXISTS taxi_fairness;
USE taxi_fairness;

-- ============================================
-- TABLE 1: Raw Taxi Trip Data
-- ============================================
DROP TABLE IF EXISTS taxi_trips_raw;

CREATE EXTERNAL TABLE taxi_trips_raw (
    VendorID INT,
    tpep_pickup_datetime TIMESTAMP,
    tpep_dropoff_datetime TIMESTAMP,
    passenger_count INT,
    trip_distance DOUBLE,
    pickup_longitude DOUBLE,
    pickup_latitude DOUBLE,
    RatecodeID INT,
    store_and_fwd_flag STRING,
    dropoff_longitude DOUBLE,
    dropoff_latitude DOUBLE,
    payment_type INT,
    fare_amount DOUBLE,
    extra DOUBLE,
    mta_tax DOUBLE,
    tip_amount DOUBLE,
    tolls_amount DOUBLE,
    improvement_surcharge DOUBLE,
    total_amount DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/taxi/raw/'
TBLPROPERTIES ('skip.header.line.count'='1');

-- ============================================
-- TABLE 2: Census Income Data
-- ============================================
DROP TABLE IF EXISTS census_income;

CREATE EXTERNAL TABLE census_income (
    zip_code STRING,
    geography STRING,
    geographic_area_name STRING,
    households DOUBLE,
    households_margin_error DOUBLE,
    median_income DOUBLE,
    median_income_margin_error DOUBLE,
    mean_income DOUBLE,
    mean_income_margin_error DOUBLE,
    year INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/census/'
TBLPROPERTIES ('skip.header.line.count'='1');

-- ============================================
-- TABLE 3: NYC ZIP Code Mapping
-- ============================================
DROP TABLE IF EXISTS nyc_zip_mapping;

CREATE EXTERNAL TABLE nyc_zip_mapping (
    zip_code STRING,
    neighborhood STRING,
    borough STRING,
    latitude DOUBLE,
    longitude DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/mapping/'
TBLPROPERTIES ('skip.header.line.count'='1');

-- ============================================
-- TABLE 4: Cleaned Taxi Data (Partitioned)
-- ============================================
DROP TABLE IF EXISTS taxi_trips_cleaned;

CREATE TABLE taxi_trips_cleaned (
    trip_id STRING,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INT,
    trip_distance DOUBLE,
    pickup_longitude DOUBLE,
    pickup_latitude DOUBLE,
    dropoff_longitude DOUBLE,
    dropoff_latitude DOUBLE,
    fare_amount DOUBLE,
    total_amount DOUBLE,
    trip_duration_minutes DOUBLE,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour BOOLEAN,
    pickup_zip STRING
)
PARTITIONED BY (year INT, month INT)
STORED AS PARQUET
LOCATION '/bigdata/processed/taxi_cleaned/';

-- ============================================
-- TABLE 5: Enriched Taxi Data with Income
-- ============================================
DROP TABLE IF EXISTS taxi_trips_enriched;

CREATE TABLE taxi_trips_enriched (
    trip_id STRING,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INT,
    trip_distance DOUBLE,
    pickup_longitude DOUBLE,
    pickup_latitude DOUBLE,
    dropoff_longitude DOUBLE,
    dropoff_latitude DOUBLE,
    fare_amount DOUBLE,
    total_amount DOUBLE,
    trip_duration_minutes DOUBLE,
    hour_of_day INT,
    day_of_week INT,
    is_rush_hour BOOLEAN,
    pickup_zip STRING,
    neighborhood STRING,
    borough STRING,
    median_income DOUBLE,
    income_category STRING
)
PARTITIONED BY (year INT, month INT)
STORED AS PARQUET
LOCATION '/bigdata/processed/taxi_enriched/';

-- ============================================
-- TABLE 6: ML Predictions
-- ============================================
DROP TABLE IF EXISTS ml_predictions;

CREATE TABLE ml_predictions (
    trip_id STRING,
    actual_fare DOUBLE,
    predicted_fare_baseline DOUBLE,
    predicted_fare_fair DOUBLE,
    income_category STRING,
    borough STRING,
    trip_distance DOUBLE
)
STORED AS PARQUET
LOCATION '/bigdata/results/predictions/';

-- ============================================
-- USEFUL QUERIES FOR ANALYSIS
-- ============================================

-- Query 1: Count trips by borough
-- SELECT borough, COUNT(*) as trip_count 
-- FROM taxi_trips_enriched 
-- GROUP BY borough 
-- ORDER BY trip_count DESC;

-- Query 2: Average fare by income category
-- SELECT income_category, 
--        AVG(fare_amount) as avg_fare,
--        AVG(trip_distance) as avg_distance,
--        COUNT(*) as trip_count
-- FROM taxi_trips_enriched
-- GROUP BY income_category;

-- Query 3: Fare per mile by income category
-- SELECT income_category,
--        AVG(fare_amount / NULLIF(trip_distance, 0)) as fare_per_mile
-- FROM taxi_trips_enriched
-- WHERE trip_distance > 0
-- GROUP BY income_category;

SHOW TABLES;
