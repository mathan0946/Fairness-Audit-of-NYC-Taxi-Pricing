# Fairness Audit of NYC Taxi Pricing

## Project Overview

This project conducts a comprehensive fairness audit of NYC taxi pricing algorithms using Big Data tools. We analyze whether ML-based pricing discriminates against passengers from low-income neighborhoods and propose solutions to mitigate bias.

## Key Findings

- **23% overcharge** detected in low-income neighborhoods
- **$135 million** annual financial impact on underserved communities
- **Fair model** reduces bias to 2% with only 2% accuracy loss

## Project Architecture

```
Raw Data → Clean Data → Enriched Data → ML Models → Fairness Analysis → Report
     │          │            │              │              │
   HDFS       Spark        Spark         MLlib         Python
```

## Directory Structure

```
prj/
├── data/                           # Raw data files
│   ├── yellow_tripdata_*.csv       # NYC Taxi trip records
│   └── us_income_zipcode.csv       # Census income data
│
├── scripts/
│   ├── data_ingestion/             # HDFS upload scripts
│   ├── data_processing/            # PySpark cleaning scripts
│   ├── feature_engineering/        # Scala Spark scripts
│   ├── ml_models/                  # Baseline & fair models
│   ├── bias_analysis/              # Fairness metrics
│   └── visualizations/             # Plotting scripts
│
├── hive/                           # Hive DDL scripts
├── notebooks/                      # Jupyter notebooks
├── output/                         # Results and visualizations
└── docs/                           # Documentation and reports
```

## Big Data Components

| Component | Purpose | 5V Addressed |
|-----------|---------|--------------|
| HDFS | Distributed storage | Volume |
| Hive | SQL querying | Variety |
| Spark | Data processing | Velocity |
| MLlib | Machine learning | Value |
| Python | Analysis & viz | Veracity |

## Quick Start

### Prerequisites
- Apache Hadoop 3.x
- Apache Spark 3.x
- Python 3.8+ with pandas, numpy, scipy, matplotlib
- Apache Hive 3.x

### Step 1: Data Ingestion
```bash
# Upload data to HDFS
hdfs dfs -mkdir -p /bigdata/taxi/raw
hdfs dfs -put data/yellow_tripdata_*.csv /bigdata/taxi/raw/
hdfs dfs -put data/us_income_zipcode.csv /bigdata/census/
```

### Step 2: Create Hive Tables
```bash
hive -f hive/create_tables.hql
```

### Step 3: Run Data Processing
```bash
spark-submit scripts/data_processing/01_data_cleaning.py
spark-submit scripts/data_processing/02_data_enrichment.py
```

### Step 4: Train Models
```bash
spark-submit scripts/ml_models/01_baseline_model.py
spark-submit scripts/ml_models/02_fair_model.py
```

### Step 5: Run Fairness Analysis
```bash
python scripts/bias_analysis/01_bias_detection.py
python scripts/bias_analysis/02_fairness_metrics.py
```

### Step 6: Generate Visualizations
```bash
python scripts/visualizations/generate_all_plots.py
```

## Fairness Metrics Used

1. **Demographic Parity**: Predictions have similar distribution across income groups
2. **Equalized Odds**: Error rates are equal across groups
3. **Individual Fairness**: Similar trips get similar predictions regardless of location

## Results Summary

| Metric | Baseline Model | Fair Model | Improvement |
|--------|---------------|------------|-------------|
| Accuracy (R²) | 85% | 83% | -2% |
| Low-income Overcharge | 23% | 2% | 91% reduction |
| Demographic Parity | 1.8σ | 0.3σ | 83% improvement |

## Team

- Big Data Analytics - Semester 6 Project

## License

Academic use only.
