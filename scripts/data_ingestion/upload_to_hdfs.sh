#!/bin/bash
# ============================================
# HDFS Data Ingestion Script
# ============================================
# This script uploads all raw data files to HDFS
# Run from the project root directory

echo "=========================================="
echo "NYC Taxi Fairness Audit - Data Ingestion"
echo "=========================================="

# Configuration
HDFS_BASE="/bigdata"
LOCAL_DATA_DIR="./data"

# Create HDFS directories
echo "[1/4] Creating HDFS directory structure..."
hdfs dfs -mkdir -p ${HDFS_BASE}/taxi/raw
hdfs dfs -mkdir -p ${HDFS_BASE}/census
hdfs dfs -mkdir -p ${HDFS_BASE}/mapping
hdfs dfs -mkdir -p ${HDFS_BASE}/processed/taxi_cleaned
hdfs dfs -mkdir -p ${HDFS_BASE}/processed/taxi_enriched
hdfs dfs -mkdir -p ${HDFS_BASE}/processed/ml_ready
hdfs dfs -mkdir -p ${HDFS_BASE}/results/baseline_predictions
hdfs dfs -mkdir -p ${HDFS_BASE}/results/fair_predictions

echo "[2/4] Uploading NYC Taxi trip data..."
for file in ${LOCAL_DATA_DIR}/yellow_tripdata_*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  Uploading: $filename"
        hdfs dfs -put -f "$file" ${HDFS_BASE}/taxi/raw/
    fi
done

echo "[3/4] Uploading Census income data..."
hdfs dfs -put -f ${LOCAL_DATA_DIR}/us_income_zipcode.csv ${HDFS_BASE}/census/

echo "[4/4] Verifying uploads..."
echo ""
echo "Taxi data files:"
hdfs dfs -ls ${HDFS_BASE}/taxi/raw/
echo ""
echo "Census data files:"
hdfs dfs -ls ${HDFS_BASE}/census/

echo ""
echo "=========================================="
echo "Data ingestion complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run Hive table creation: hive -f hive/create_tables.hql"
echo "2. Run data cleaning: spark-submit scripts/data_processing/01_data_cleaning.py"
