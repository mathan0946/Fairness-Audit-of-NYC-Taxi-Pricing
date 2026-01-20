# Fairness Audit of NYC Taxi Pricing

## Project Overview

This project conducts a comprehensive fairness audit of NYC taxi pricing algorithms using Big Data tools. We analyze whether ML-based pricing discriminates against passengers from low-income neighborhoods and propose solutions to mitigate bias.

**Academic Project**: Big Data Analytics - Semester 6  
**Repository**: https://github.com/mathan0946/Fairness-Audit-of-NYC-Taxi-Pricing.git

## Implementation Status

### ✅ Completed
- [x] Project structure and scaffolding
- [x] Git repository initialization
- [x] Docker-based Hadoop HDFS cluster setup (BDE stack)
- [x] Docker-based Apache Spark environment
- [x] Data ingestion to HDFS (5.3 GB - 4 files)
- [x] PySpark data cleaning pipeline (46M records processed)
- [x] All Python scripts (data processing, ML models, bias analysis)
- [x] Scala feature engineering script
- [x] Hive DDL scripts
- [x] Jupyter notebook for analysis
- [x] Project documentation

### 🔄 In Progress
- [ ] Data enrichment with census income data
- [ ] Feature engineering
- [ ] ML model training (baseline and fair models)
- [ ] Bias detection analysis
- [ ] Fairness metrics calculation
- [ ] Visualization generation

### 📊 Expected Key Findings
- **23% overcharge** detected in low-income neighborhoods (target)
- **$135 million** annual financial impact estimate
- **Fair model** to reduce bias to <5% with minimal accuracy loss

## Data Pipeline Progress

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data   │ ──> │ Data Cleanup │ ──> │ Enrichment    │ ──> │  ML Models   │ ──> │   Analysis   │
│   (HDFS)    │     │   (PySpark)  │     │  (PySpark)    │     │   (MLlib)    │     │   (Python)   │
└─────────────┘     └──────────────┘     └───────────────┘     └──────────────┘     └──────────────┘
      ✅                   ✅                     🔄                    📝                  📝

  5.3 GB              46M records           Joining with          Training            Bias Detection
  4 files             18 features           census data           RF models           + Metrics
```

## Technical Details

### Data Schema (After Cleaning)
```
root
 |-- trip_id: string                    # Unique identifier
 |-- tpep_pickup_datetime: timestamp    # Pickup time
 |-- tpep_dropoff_datetime: timestamp   # Dropoff time
 |-- passenger_count: integer           # Number of passengers
 |-- trip_distance: double              # Distance in miles
 |-- pickup_longitude: double           # Pickup GPS
 |-- pickup_latitude: double            # Pickup GPS
 |-- dropoff_longitude: double          # Dropoff GPS
 |-- dropoff_latitude: double           # Dropoff GPS
 |-- fare_amount: double                # Base fare
 |-- total_amount: double               # Total fare paid
 |-- trip_duration_minutes: double      # Calculated duration
 |-- hour_of_day: integer               # 0-23
 |-- day_of_week: integer               # 1-7
 |-- is_rush_hour: boolean              # 7-9 AM, 5-7 PM
 |-- is_weekend: boolean                # Saturday/Sunday
 |-- is_night: boolean                  # 10 PM - 5 AM
 |-- fare_per_mile: double              # Efficiency metric
 |-- pickup_zone: string                # Approximate ZIP grid
```

### Data Quality Filters Applied
| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Max Fare | $500 | Remove extreme outliers |
| Max Distance | 100 miles | NYC metro area limit |
| Max Duration | 300 minutes | Remove stalled trips |
| Max Speed | 100 mph | Physical limit |
| Min Fare | $2.50 | NYC base fare |

## Project Structure

```
prj/
├── .git/                           # Git repository
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Master pipeline runner
├── docker-compose.yml              # Docker orchestration (created)
├── hadoop.env                      # Hadoop environment config
│
├── data/                           # Raw data files (5.4 GB CSV)
│   ├── yellow_tripdata_2016-01.csv # Jan 2016 (1.7 GB)
│   ├── yellow_tripdata_2016-02.csv # Feb 2016 (1.8 GB)
│   ├── yellow_tripdata_2016-03.csv # Mar 2016 (1.9 GB)
│   └── us_income_zipcode.csv       # Census data (200 MB)
│
├── output/                         # Processing results
│   └── processed/
│       └── taxi_cleaned/           # Cleaned data (1.3 GB Parquet) ✅
│
├── scripts/
│   ├── data_ingestion/
│   │   └── upload_to_hdfs.sh       # HDFS upload script ✅
│   ├── data_processing/
│   │   ├── 01_data_cleaning.py     # PySpark cleaning ✅
│   │   └── 02_data_enrichment.py   # Income join 🔄
│   ├── feature_engineering/
│   │   ├── FeatureEngineering.scala # Scala Spark features
│   │   └── feature_engineering.py   # Python alternative
│   ├── ml_models/
│   │   ├── 01_baseline_model.py    # Biased model
│   │   └── 02_fair_model.py        # Fair model
│   ├── bias_analysis/
│   │   ├── 01_bias_detection.py    # Statistical tests
│   │   └── 02_fairness_metrics.py  # Academic metrics
│   └── visualizations/
│       └── generate_all_plots.py   # 6 charts + Tableau CSV
│
├── hive/
│   └── create_tables.hql           # Hive DDL (6 tables)
│
├── notebooks/
│   └── fairness_audit_analysis.ipynb # Jupyter notebook
│
└── docs/
    └── project_report_outline.md   # 20-page report template
```

## Big Data Components

| Component | Purpose | 5V Addressed |
|-----------|---------|--------------|
| HDFS | Distributed storage | Volume |
| Hive | SQL querying | Variety |
| Spark | Data processing | Velocity |
| MLlib | Machine learning | Value |
| Python | Analysis & viz | Veracity |

## Environment Setup

### Current Infrastructure
- **HDFS**: BDE Hadoop 3.2.1 (Docker containers: namenode, datanode)
- **YARN**: ResourceManager + NodeManager for cluster management
- **Spark**: Apache Spark 3.5.1 (Docker container)
- **Python**: 3.12 with virtual environment
- **MongoDB**: 6.0 (auxiliary database)

### Docker Containers Running
```bash
# Hadoop Cluster
namenode            - HDFS NameNode (Web UI: http://localhost:9870)
datanode            - HDFS DataNode
resourcemanager     - YARN ResourceManager (Web UI: http://localhost:8088)
nodemanager         - YARN NodeManager
historyserver       - Hadoop History Server

# Processing
spark               - Apache Spark 3.5.1 (UI: http://localhost:8080)
```

### Prerequisites
- Docker Desktop (running)
- Python 3.12+
- Git

## Implementation Steps Completed

### Step 1: Data Ingestion ✅
```bash
# Data uploaded to HDFS at:
# /bigdata/taxi/raw/ (3 files, 5.1 GB)
# /bigdata/census/ (1 file, 190.5 MB)

# Verify:
docker exec namenode hdfs dfs -ls -h /bigdata/taxi/raw/
docker exec namenode hdfs dfs -ls -h /bigdata/census/
```

**Result**: 
- `yellow_tripdata_2016-01.csv` (1.6 GB) ✅
- `yellow_tripdata_2016-02.csv` (1.7 GB) ✅  
- `yellow_tripdata_2016-03.csv` (1.8 GB) ✅
- `us_income_zipcode.csv` (190.5 MB) ✅

### Step 2: Data Cleaning ✅
```bash
# Executed PySpark data cleaning on 47M records
docker exec spark /opt/spark/bin/spark-submit \
  --master local[*] \
  --driver-memory 4g \
  /opt/spark/work-dir/scripts/data_processing/01_data_cleaning.py
```

**Result**:
- **Initial records**: 47,248,845
- **Final records**: 46,096,114
- **Records removed**: 1,152,731 (2.4% - nulls & outliers)
- **Output**: 1.3 GB Parquet (56 partitions)
- **Location**: `output/output/processed/taxi_cleaned/`

**Data Transformations**:
- ✅ Removed null values
- ✅ Removed outliers (fare >$500, distance >100 miles, duration >5 hours)
- ✅ Added derived fields:
  - `trip_id` (unique identifier)
  - `trip_duration_minutes` (calculated)
  - `hour_of_day`, `day_of_week` (temporal)
  - `is_rush_hour`, `is_weekend`, `is_night` (boolean flags)
  - `fare_per_mile` (price efficiency metric)
  - `pickup_zone` (approximate ZIP code grid)

### Step 3: Data Enrichment 🔄 (Next)
```bash
# Join cleaned taxi data with census income data
spark-submit scripts/data_processing/02_data_enrichment.py
```

### Step 4: Feature Engineering 🔄 (Pending)
```bash
# Scala Spark feature engineering
spark-submit scripts/feature_engineering/FeatureEngineering.scala
```

### Step 5: Train Models 🔄 (Pending)
```bash
# Baseline model (with location bias)
spark-submit scripts/ml_models/01_baseline_model.py

# Fair model (without location features)
spark-submit scripts/ml_models/02_fair_model.py
```

### Step 6: Bias Analysis 🔄 (Pending)
```bash
python scripts/bias_analysis/01_bias_detection.py
python scripts/bias_analysis/02_fairness_metrics.py
```

### Step 7: Generate Visualizations 🔄 (Pending)
```bash
python scripts/visualizations/generate_all_plots.py
```

## Fairness Metrics Used

1. **Demographic Parity**: Predictions have similar distribution across income groups
2. **Equalized Odds**: Error rates are equal across groups
3. **Individual Fairness**: Similar trips get similar predictions regardless of location

## Results Summary (Target Metrics)

| Metric | Baseline Model | Fair Model | Improvement |
|--------|---------------|------------|-------------|
| Accuracy (R²) | 85% (target) | 83% (target) | -2% |
| Low-income Overcharge | 23% (to detect) | 2% (target) | 91% reduction |
| Demographic Parity | 1.8σ (expected) | 0.3σ (target) | 83% improvement |

*Note: These are target metrics - actual results pending model training*

## Development Timeline

- **Week 1-2**: Project setup, data collection ✅
- **Week 3-4**: Data ingestion to HDFS ✅
- **Week 5**: Data cleaning with Spark ✅
- **Week 6**: Data enrichment 🔄
- **Week 7-8**: ML model training 📝
- **Week 9**: Bias analysis 📝
- **Week 10**: Visualization & reporting 📝

## Quick Commands

```bash
# Start Hadoop cluster
docker start namenode datanode resourcemanager nodemanager historyserver

# Check HDFS data
docker exec namenode hdfs dfs -ls -h /bigdata/

# Run Spark job
docker exec spark /opt/spark/bin/spark-submit \
  --master local[*] \
  /opt/spark/work-dir/scripts/data_processing/01_data_cleaning.py

# Access web UIs
# HDFS: http://localhost:9870
# YARN: http://localhost:8088  
# Spark: http://localhost:8080

# Commit changes
git add .
git commit -m "Your message"
git push origin main
```

## Technologies Used

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Storage | HDFS | 3.2.1 | Distributed file system |
| Processing | Apache Spark | 3.5.1 | Data processing |
| Container | Docker | 28.3.2 | Infrastructure |
| Language | Python | 3.12 | Analysis & ML |
| Language | Scala | 2.12 | Feature engineering |
| Notebook | Jupyter | Latest | Interactive analysis |
| Database | MongoDB | 6.0 | Auxiliary storage |
| VCS | Git/GitHub | - | Version control |

## Big Data - 5 V's Analysis

| V | Aspect | Implementation |
|---|--------|----------------|
| **Volume** | 5.3 GB raw data (46M records) | Distributed storage on HDFS |
| **Velocity** | Real-time taxi trip streaming | Spark parallel processing |
| **Variety** | CSV, Parquet, Census data | Multi-format support |
| **Veracity** | Data quality issues | Cleaning pipeline (2.4% removed) |
| **Value** | Business insights | Fairness metrics, $135M impact |

## Contact & Contributors

**Project Team**: Big Data Analytics - Semester 6  
**Institution**: [Your Institution Name]  
**GitHub**: https://github.com/mathan0946/Fairness-Audit-of-NYC-Taxi-Pricing

## License

Academic use only. Not for commercial distribution.

---

**Last Updated**: January 20, 2026  
**Project Status**: 🔄 In Progress (Data cleaning completed, enrichment pending)
