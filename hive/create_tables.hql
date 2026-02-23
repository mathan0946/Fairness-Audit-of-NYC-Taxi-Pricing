-- ============================================================
-- HIVE DDL SCRIPT: NYC Taxi Fairness Audit
-- ============================================================
-- Big Data Framework: Apache Hive (HiveQL)
-- Purpose: Define structured schema over HDFS data
-- Run:     beeline -u jdbc:hive2://localhost:10000 -f hive/create_tables.hql
--          OR  spark-sql -f hive/create_tables.hql   (via Spark's Hive support)
-- ============================================================

-- ============================================================
-- DATABASE
-- ============================================================
CREATE DATABASE IF NOT EXISTS taxi_fairness
  COMMENT 'NYC Taxi Pricing Fairness Audit – Big Data Analytics Project';
USE taxi_fairness;

-- ============================================================
-- TABLE 1: Raw Taxi Trip Data  (EXTERNAL over HDFS CSV files)
-- ============================================================
DROP TABLE IF EXISTS taxi_trips_raw;

CREATE EXTERNAL TABLE taxi_trips_raw (
    VendorID              INT       COMMENT 'TPEP provider (1=CMT, 2=VTS)',
    tpep_pickup_datetime  TIMESTAMP COMMENT 'Meter engagement timestamp',
    tpep_dropoff_datetime TIMESTAMP COMMENT 'Meter disengagement timestamp',
    passenger_count       INT       COMMENT 'Number of passengers',
    trip_distance         DOUBLE    COMMENT 'Trip distance in miles',
    pickup_longitude      DOUBLE    COMMENT 'Pickup GPS longitude',
    pickup_latitude       DOUBLE    COMMENT 'Pickup GPS latitude',
    RatecodeID            INT       COMMENT 'Rate code (1=Standard, 2=JFK…)',
    store_and_fwd_flag    STRING    COMMENT 'Y = store-and-forward trip',
    dropoff_longitude     DOUBLE    COMMENT 'Dropoff GPS longitude',
    dropoff_latitude      DOUBLE    COMMENT 'Dropoff GPS latitude',
    payment_type          INT       COMMENT '1=Credit, 2=Cash, 3=No charge…',
    fare_amount           DOUBLE    COMMENT 'Time-and-distance fare (USD)',
    extra                 DOUBLE    COMMENT 'Misc extras & surcharges',
    mta_tax               DOUBLE    COMMENT 'MTA tax ($0.50)',
    tip_amount            DOUBLE    COMMENT 'Tip amount (credit only)',
    tolls_amount          DOUBLE    COMMENT 'Total tolls paid',
    improvement_surcharge DOUBLE    COMMENT 'Improvement surcharge ($0.30)',
    total_amount          DOUBLE    COMMENT 'Total charged to passenger'
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/taxi/raw/'
TBLPROPERTIES (
    'skip.header.line.count' = '1',
    'serialization.null.format' = ''
);

-- ============================================================
-- TABLE 2: US Census Income Data  (EXTERNAL over HDFS CSV)
-- ============================================================
DROP TABLE IF EXISTS census_income;

CREATE EXTERNAL TABLE census_income (
    zip_code                   STRING  COMMENT 'Five-digit ZIP code',
    geography                  STRING  COMMENT 'Geography identifier',
    geographic_area_name       STRING  COMMENT 'Human-readable area name',
    households                 DOUBLE  COMMENT 'Total number of households',
    households_margin_error    DOUBLE  COMMENT 'Margin of error – households',
    median_income              DOUBLE  COMMENT 'Median household income (USD)',
    median_income_margin_error DOUBLE  COMMENT 'Margin of error – median income',
    mean_income                DOUBLE  COMMENT 'Mean household income (USD)',
    mean_income_margin_error   DOUBLE  COMMENT 'Margin of error – mean income',
    year                       INT     COMMENT 'Census survey year'
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/census/'
TBLPROPERTIES (
    'skip.header.line.count' = '1',
    'serialization.null.format' = ''
);

-- ============================================================
-- TABLE 3: Cleaned Taxi Data  (Parquet)
-- Written by Spark DataCleaning stage
-- ============================================================
DROP TABLE IF EXISTS taxi_trips_cleaned;

CREATE EXTERNAL TABLE taxi_trips_cleaned (
    trip_id                STRING    COMMENT 'Unique trip identifier',
    tpep_pickup_datetime   TIMESTAMP COMMENT 'Pickup timestamp',
    tpep_dropoff_datetime  TIMESTAMP COMMENT 'Dropoff timestamp',
    passenger_count        INT       COMMENT 'Passengers (1-9)',
    trip_distance          DOUBLE    COMMENT 'Distance in miles (0-100)',
    pickup_longitude       DOUBLE    COMMENT 'Pickup longitude',
    pickup_latitude        DOUBLE    COMMENT 'Pickup latitude',
    dropoff_longitude      DOUBLE    COMMENT 'Dropoff longitude',
    dropoff_latitude       DOUBLE    COMMENT 'Dropoff latitude',
    fare_amount            DOUBLE    COMMENT 'Base fare (USD)',
    total_amount           DOUBLE    COMMENT 'Total paid (USD)',
    trip_duration_minutes  DOUBLE    COMMENT 'Trip duration in minutes',
    speed_mph              DOUBLE    COMMENT 'Trip average speed (mph)',
    hour_of_day            INT       COMMENT 'Hour 0-23',
    day_of_week            INT       COMMENT 'Day 1-7',
    is_rush_hour           BOOLEAN   COMMENT '7-9 AM or 5-7 PM weekdays',
    is_weekend             BOOLEAN   COMMENT 'Saturday or Sunday',
    is_night               BOOLEAN   COMMENT '10 PM - 6 AM',
    fare_per_mile          DOUBLE    COMMENT 'Fare divided by distance',
    pickup_zone            STRING    COMMENT 'Grid-based zone key'
)
STORED AS PARQUET
LOCATION '/bigdata/processed/taxi_cleaned/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ============================================================
-- TABLE 4: Enriched Taxi Data (Parquet)
-- Written by Spark DataEnrichment stage
-- ============================================================
DROP TABLE IF EXISTS taxi_trips_enriched;

CREATE EXTERNAL TABLE taxi_trips_enriched (
    trip_id                STRING    COMMENT 'Unique trip identifier',
    tpep_pickup_datetime   TIMESTAMP COMMENT 'Pickup timestamp',
    tpep_dropoff_datetime  TIMESTAMP COMMENT 'Dropoff timestamp',
    passenger_count        INT       COMMENT 'Passengers',
    trip_distance          DOUBLE    COMMENT 'Distance in miles',
    pickup_longitude       DOUBLE    COMMENT 'Pickup longitude',
    pickup_latitude        DOUBLE    COMMENT 'Pickup latitude',
    dropoff_longitude      DOUBLE    COMMENT 'Dropoff longitude',
    dropoff_latitude       DOUBLE    COMMENT 'Dropoff latitude',
    fare_amount            DOUBLE    COMMENT 'Fare (USD)',
    total_amount           DOUBLE    COMMENT 'Total (USD)',
    trip_duration_minutes  DOUBLE    COMMENT 'Duration (min)',
    speed_mph              DOUBLE    COMMENT 'Speed (mph)',
    hour_of_day            INT       COMMENT 'Hour 0-23',
    day_of_week            INT       COMMENT 'Day 1-7',
    is_rush_hour           BOOLEAN   COMMENT 'Rush-hour flag',
    is_weekend             BOOLEAN   COMMENT 'Weekend flag',
    is_night               BOOLEAN   COMMENT 'Night flag',
    fare_per_mile          DOUBLE    COMMENT 'Fare / distance',
    pickup_zip             STRING    COMMENT 'Nearest NYC ZIP code',
    neighborhood           STRING    COMMENT 'Neighborhood name',
    borough                STRING    COMMENT 'NYC borough',
    median_income          DOUBLE    COMMENT 'ZIP median household income',
    income_category        STRING    COMMENT 'high / medium / low'
)
STORED AS PARQUET
LOCATION '/bigdata/processed/taxi_enriched/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ============================================================
-- TABLE 5: ML Predictions (Parquet)
-- Written by Spark MLPipeline stage
-- ============================================================
DROP TABLE IF EXISTS ml_predictions;

CREATE EXTERNAL TABLE ml_predictions (
    trip_id                   STRING COMMENT 'Trip identifier',
    fare_amount               DOUBLE COMMENT 'Actual fare',
    predicted_fare_baseline   DOUBLE COMMENT 'Baseline model prediction',
    predicted_fare_fair       DOUBLE COMMENT 'Fair model prediction',
    baseline_error            DOUBLE COMMENT 'Baseline pred – actual',
    fair_error                DOUBLE COMMENT 'Fair pred – actual',
    baseline_abs_error        DOUBLE COMMENT '|baseline_error|',
    fair_abs_error            DOUBLE COMMENT '|fair_error|',
    income_category           STRING COMMENT 'high / medium / low',
    borough                   STRING COMMENT 'NYC borough',
    neighborhood              STRING COMMENT 'Neighborhood',
    trip_distance             DOUBLE COMMENT 'Distance (miles)',
    trip_duration_minutes     DOUBLE COMMENT 'Duration (min)',
    median_income             DOUBLE COMMENT 'Area median income'
)
STORED AS PARQUET
LOCATION '/bigdata/results/predictions/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- ============================================================
-- TABLE 6: Bias Summary by Income Category  (CSV)
-- ============================================================
DROP TABLE IF EXISTS bias_summary;

CREATE EXTERNAL TABLE bias_summary (
    income_category          STRING COMMENT 'Income group',
    trip_count               BIGINT COMMENT 'Number of trips',
    avg_distance             DOUBLE COMMENT 'Mean trip distance',
    avg_actual_fare          DOUBLE COMMENT 'Mean actual fare',
    baseline_predicted       DOUBLE COMMENT 'Baseline mean prediction',
    fair_predicted           DOUBLE COMMENT 'Fair mean prediction',
    baseline_avg_error       DOUBLE COMMENT 'Baseline mean error',
    fair_avg_error           DOUBLE COMMENT 'Fair mean error',
    baseline_overcharge_pct  DOUBLE COMMENT 'Baseline overcharge %',
    fair_overcharge_pct      DOUBLE COMMENT 'Fair overcharge %',
    avg_area_income          DOUBLE COMMENT 'Avg area median income'
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/results/bias_summary_csv/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ============================================================
-- TABLE 7: Borough-level Bias  (CSV)
-- ============================================================
DROP TABLE IF EXISTS borough_bias;

CREATE EXTERNAL TABLE borough_bias (
    borough                  STRING COMMENT 'NYC borough',
    trip_count               BIGINT COMMENT 'Number of trips',
    avg_income               DOUBLE COMMENT 'Borough avg income',
    avg_actual_fare          DOUBLE COMMENT 'Mean actual fare',
    baseline_predicted       DOUBLE COMMENT 'Baseline mean pred',
    fair_predicted           DOUBLE COMMENT 'Fair mean pred',
    baseline_overcharge_pct  DOUBLE COMMENT 'Baseline OC %',
    fair_overcharge_pct      DOUBLE COMMENT 'Fair OC %'
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/bigdata/results/borough_bias_csv/'
TBLPROPERTIES ('skip.header.line.count' = '1');

-- ============================================================
-- ANALYTICAL VIEWS (run after pipeline completes)
-- ============================================================

-- View: Trip volume and avg fare by income category
CREATE VIEW IF NOT EXISTS v_income_fare_summary AS
SELECT income_category,
       COUNT(*)                      AS trip_count,
       ROUND(AVG(trip_distance), 2)  AS avg_distance,
       ROUND(AVG(fare_amount), 2)    AS avg_fare,
       ROUND(AVG(fare_per_mile), 2)  AS avg_fare_per_mile,
       ROUND(AVG(median_income), 0)  AS avg_median_income
FROM   taxi_trips_enriched
GROUP  BY income_category;

-- View: Borough-level trip statistics
CREATE VIEW IF NOT EXISTS v_borough_stats AS
SELECT borough,
       COUNT(*)                      AS trip_count,
       ROUND(AVG(fare_amount), 2)    AS avg_fare,
       ROUND(AVG(trip_distance), 2)  AS avg_distance,
       ROUND(AVG(median_income), 0)  AS avg_income
FROM   taxi_trips_enriched
GROUP  BY borough;

-- View: Hourly demand patterns
CREATE VIEW IF NOT EXISTS v_hourly_demand AS
SELECT hour_of_day,
       income_category,
       COUNT(*)                      AS trip_count,
       ROUND(AVG(fare_amount), 2)    AS avg_fare
FROM   taxi_trips_enriched
GROUP  BY hour_of_day, income_category;

-- View: Model comparison (bias by income)
CREATE VIEW IF NOT EXISTS v_bias_comparison AS
SELECT income_category,
       COUNT(*)                                                            AS trips,
       ROUND(AVG(fare_amount), 2)                                         AS avg_actual,
       ROUND(AVG(predicted_fare_baseline), 2)                              AS baseline_avg,
       ROUND(AVG(predicted_fare_fair), 2)                                  AS fair_avg,
       ROUND((AVG(predicted_fare_baseline) - AVG(fare_amount))
             / AVG(fare_amount) * 100, 1)                                  AS baseline_oc_pct,
       ROUND((AVG(predicted_fare_fair) - AVG(fare_amount))
             / AVG(fare_amount) * 100, 1)                                  AS fair_oc_pct
FROM   ml_predictions
GROUP  BY income_category;

-- ============================================================
-- USEFUL AD-HOC QUERIES (commented — copy-paste as needed)
-- ============================================================

-- 1. Trip count by borough
-- SELECT borough, COUNT(*) AS trip_count
-- FROM taxi_trips_enriched GROUP BY borough ORDER BY trip_count DESC;

-- 2. Average fare by income category
-- SELECT income_category, AVG(fare_amount) AS avg_fare, COUNT(*) AS cnt
-- FROM taxi_trips_enriched GROUP BY income_category ORDER BY income_category;

-- 3. Fare per mile by income category (controlled analysis)
-- SELECT income_category,
--        AVG(fare_amount / NULLIF(trip_distance, 0)) AS fare_per_mile
-- FROM taxi_trips_enriched WHERE trip_distance > 0
-- GROUP BY income_category;

-- 4. Overcharge comparison for similar-distance trips (~5 mi)
-- SELECT income_category,
--        COUNT(*) AS trips,
--        ROUND(AVG(predicted_fare_baseline), 2) AS baseline_pred,
--        ROUND(AVG(predicted_fare_fair), 2)     AS fair_pred,
--        ROUND(AVG(fare_amount), 2)             AS actual
-- FROM ml_predictions
-- WHERE trip_distance BETWEEN 4.5 AND 5.5
-- GROUP BY income_category ORDER BY income_category;

-- 5. Top 10 neighborhoods by overcharge (baseline)
-- SELECT neighborhood, borough,
--        COUNT(*) AS trips,
--        ROUND(AVG(predicted_fare_baseline - fare_amount), 2) AS avg_overcharge
-- FROM ml_predictions
-- GROUP BY neighborhood, borough
-- ORDER BY avg_overcharge DESC LIMIT 10;

SHOW TABLES;
