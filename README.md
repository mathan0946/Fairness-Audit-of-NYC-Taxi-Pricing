# Fairness Audit of NYC Taxi Pricing

**Detecting and Mitigating Algorithmic Discrimination Using Big Data Frameworks**

> Semester 6 – Big Data Analytics Project

---

## Project Overview

This project investigates whether machine-learning pricing models for NYC taxi trips exhibit **algorithmic bias** against passengers in low-income neighborhoods. We train two Random Forest regressors — a **Baseline** model (that includes median household income as a feature) and a **Fair** model (that excludes it) — then quantify the bias using three academic fairness metrics and estimate annual financial impact.

### Key Findings (Actual Pipeline Results)

| Metric | Baseline Model | Fair Model | Delta |
|---|---|---|---|
| R² accuracy | **94.85%** | **94.82%** | only 0.03 pp loss |
| RMSE | $2.23 | $2.24 | — |
| MAE | $0.67 | $0.66 | — |
| Low-income overcharge | +0.3% | −0.2% | **bias eliminated** |
| Income feature importance | 3.67% | removed | — |
| Demographic Parity σ | 3.11 | 3.05 | 2.1% improvement |
| Annual financial impact | ~$165K | eliminated | — |
| Records processed | 46M cleaned → 41M enriched → 2M sampled for ML | | |

---

## Big Data Frameworks Used

| Framework | Version | Role |
|---|---|---|
| **Apache Spark** | 4.1.1 | Distributed data processing, ML Pipeline |
| **Spark SQL** | 4.1.1 | Structured queries, Parquet I/O |
| **Spark MLlib** | 4.1.1 | Random Forest, VectorAssembler, StandardScaler, RegressionEvaluator |
| **Apache Hive** | 4.0.1 | DDL schema over HDFS, analytical views |
| **Apache Kafka** | 7.6 (Confluent) | Real-time taxi trip stream simulation |
| **Hadoop HDFS** | 3.2.1 (bde2020) | Distributed storage (Docker cluster) |
| **Scala** | 2.13.17 | All Spark processing scripts |
| **Python** | 3.12 | scipy stats, matplotlib/seaborn visualisations |
| **Docker** | — | Hadoop + Hive + Kafka cluster |

---

## Repository Structure

```
prj/
├── build.sbt                         # Scala/Spark build config
├── docker-compose.yml                # Hadoop + Hive cluster
├── hadoop.env                        # Hadoop environment vars
├── run_pipeline.bat                  # Master pipeline runner (5 stages)
├── requirements.txt                  # Python dependencies
│
├── data/                             # Raw datasets
│   ├── yellow_tripdata_2016-01.csv
│   ├── yellow_tripdata_2016-02.csv
│   ├── yellow_tripdata_2016-03.csv
│   └── us_income_zipcode.csv
│
├── hive/
│   └── create_tables.hql             # 7 Hive tables + 4 analytical views
│
├── scripts/
│   ├── scala/
│   │   ├── 01_DataCleaning.scala     # Stage 1 – Clean 47M+ raw records
│   │   ├── 02_DataEnrichment.scala   # Stage 2 – Census income join
│   │   └── 03_MLPipeline.scala       # Stage 3 – MLlib RF + bias analysis
│   ├── bias_analysis/
│   │   ├── 01_bias_detection.py      # Stage 4A – t-tests, Cohen's d
│   │   └── 02_fairness_metrics.py    # Stage 4B – DP, EO, IF metrics
│   ├── kafka/
│   │   ├── kafka_producer.py          # Stream taxi CSV → Kafka topic
│   │   └── kafka_consumer.py          # Consume Kafka → JSON-lines files
│   └── visualizations/
│       └── generate_all_plots.py     # Stage 5 – 6 charts + Tableau export
│
├── output/                           # All pipeline outputs
│   ├── processed/
│   │   ├── taxi_cleaned/             # Stage 1 Parquet
│   │   └── taxi_enriched/            # Stage 2 Parquet
│   ├── models/
│   │   ├── baseline_model/           # MLlib RF (with income)
│   │   └── fair_model/               # MLlib RF (without income)
│   ├── results/
│   │   ├── predictions/              # Parquet – both models
│   │   ├── bias_analysis/            # CSV + JSON reports
│   │   ├── fairness_metrics/         # JSON report
│   │   └── model_comparison/         # CSV for Tableau
│   └── visualizations/               # 6 PNG charts + tableau_data.csv
│
├── notebooks/
│   └── fairness_audit_analysis.ipynb
│
├── docs/
│   ├── project_report_outline.md
│   └── SCALA_SPARK_GUIDE.md
│
└── spark-4.1.1-bin-hadoop3/          # Local Spark installation
```

---

## Pipeline Stages

### Stage 1 – Data Cleaning (Scala / Spark SQL)
**Script:** `scripts/scala/01_DataCleaning.scala`

- Reads 3 months raw CSV (~5.3 GB, 47M+ rows)
- Explicit schema enforcement (no schema inference)
- Outlier filtering: fare ($2.50–$500), distance (0–100 mi), duration (0–300 min), speed (≤ 100 mph), NYC bounding box
- Derived columns: `trip_duration_minutes`, `speed_mph`, `hour_of_day`, `day_of_week`, `is_rush_hour`, `is_weekend`, `is_night`, `fare_per_mile`, `pickup_zone`
- Deterministic `trip_id` via SHA-256 hash
- Output: Snappy-compressed Parquet → `output/processed/taxi_cleaned/`

### Stage 2 – Data Enrichment (Scala / Spark SQL + Broadcast Join)
**Script:** `scripts/scala/02_DataEnrichment.scala`

- 60+ NYC geographic bounding-box zones (all 5 boroughs)
- UDF-based GPS → ZIP code mapping
- Broadcast join with US Census income data
- Income categorisation: **high** (≥ $75K), **medium** (≥ $45K), **low** (< $45K)
- Fallback to $60,000 NYC median for unmatched ZIPs
- Census data deduplicated by ZIP (avg income per unique ZIP)
- Output: 20-partition Snappy Parquet → `output/processed/taxi_enriched/`

### Stage 3 – ML Pipeline + Bias Analysis (Scala / Spark MLlib)
**Script:** `scripts/scala/03_MLPipeline.scala`

- **Spark ML Pipeline**: VectorAssembler → StandardScaler (both feature sets)
- **Baseline Random Forest**: 50 trees, max-depth 8, features include `median_income`
- **Fair Random Forest**: same hyper-params, features exclude income/location
- **5% stratified sample** (~2M records) for tractable local training
- RegressionEvaluator: R², RMSE, MAE
- Feature importance ranking
- Bias analysis by income category + controlled-distance analysis
- Demographic Parity (σ of group-mean predictions)
- Equalized Odds (σ of group-level RMSE)
- Financial impact projection (165M annual NYC trips)
- Saves: models, predictions Parquet, bias summary CSV, borough bias CSV

### Stage 4 – Statistical Analysis (Python / scipy)
**Scripts:** `scripts/bias_analysis/01_bias_detection.py`, `02_fairness_metrics.py`

- Welch's t-test (low vs high income errors)
- Cohen's d effect size
- Individual Fairness (Lipschitz condition – paired same-distance trips)
- Comprehensive JSON reports

### Stage 5 – Visualizations (Python / matplotlib + seaborn)
**Script:** `scripts/visualizations/generate_all_plots.py`

- 6 publication-quality charts (300 DPI)
- Tableau-ready CSV export with geographic coordinates

---

## How to Run

### Prerequisites
- Java JDK 21
- Apache Spark 4.1.1 (included in repo)
- Python 3.12 with virtual environment
- Docker Desktop (for HDFS/Hive cluster, optional for local mode)

### Quick Start (Local Mode)
```bat
REM 1. Activate Python venv
call venv\Scripts\activate

REM 2. Install Python dependencies
pip install -r requirements.txt

REM 3. Run full pipeline
run_pipeline.bat
```

### Individual Stages
```bat
REM Stage 1 – Data Cleaning
spark-shell --driver-memory 8g -i scripts/scala/01_DataCleaning.scala

REM Stage 2 – Data Enrichment
spark-shell --driver-memory 8g -i scripts/scala/02_DataEnrichment.scala

REM Stage 3 – ML Pipeline
spark-shell --driver-memory 8g -i scripts/scala/03_MLPipeline.scala

REM Stage 4 – Bias Analysis
python scripts/bias_analysis/01_bias_detection.py
python scripts/bias_analysis/02_fairness_metrics.py

REM Stage 5 – Visualizations
python scripts/visualizations/generate_all_plots.py
```

### Docker Services (HDFS + Hive + Kafka)
```bash
# 1. Start HDFS cluster (pre-existing bde2020 containers)
docker start namenode datanode resourcemanager nodemanager historyserver

# 2. Start Hive + Kafka (compose manages these)
cd "N:\CAI\sem 6\BGA\prj"
docker compose up -d

# 3. Create Hive tables (wait ~90s for HiveServer2 to start)
docker exec hiveserver2 beeline -u "jdbc:hive2://localhost:10000" -f /hive_scripts/create_tables.hql

# 4. Kafka streaming simulation
python scripts/kafka/kafka_producer.py --rate 500 --limit 5000
python scripts/kafka/kafka_consumer.py --output output/kafka_ingest
```

### Service Endpoints
| Service | URL |
|---|---|
| HDFS NameNode UI | http://localhost:9870 |
| YARN Resource Manager | http://localhost:8088 |
| HiveServer2 (Beeline/JDBC) | jdbc:hive2://localhost:10000 |
| HiveServer2 Web UI | http://localhost:10002 |
| Kafka Broker | localhost:9092 |

---

## Data Sources

| Dataset | Records | Size | Source |
|---|---|---|---|
| NYC Yellow Taxi (Jan–Mar 2016) | ~47M | 5.3 GB | NYC TLC |
| US Census Income by ZIP | 33K+ | 1.8 MB | US Census Bureau |

---

## Technology Stack Summary

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  matplotlib / seaborn (6 charts)  │  Tableau (geo map)  │
├─────────────────────────────────────────────────────────┤
│                    ANALYSIS LAYER                        │
│  Python scipy (t-tests)  │  Spark MLlib (Random Forest)  │
├─────────────────────────────────────────────────────────┤
│                    PROCESSING LAYER                      │
│  Spark SQL  │  Spark ML Pipeline  │  Scala 2.13 UDFs    │
├─────────────────────────────────────────────────────────┤
│                    SCHEMA LAYER                          │
│  Apache Hive (DDL, views, queries)                      │
├─────────────────────────────────────────────────────────┤
│                    STREAMING LAYER                       │
│  Apache Kafka (real-time trip simulation)                │
├─────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                         │
│  Hadoop HDFS  │  Parquet (Snappy)  │  CSV                │
├─────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE                        │
│  Docker (Hadoop + Hive + Kafka)  │  Spark 4.1.1         │
└─────────────────────────────────────────────────────────┘
```

---

## Authors

Big Data Analytics – Semester 6 Project
