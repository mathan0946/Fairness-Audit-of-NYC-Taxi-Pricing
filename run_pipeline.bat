@echo off
REM ============================================================
REM  NYC Taxi Fairness Audit – Full Pipeline Runner
REM ============================================================
REM  Big Data Frameworks:
REM    Apache Spark 4.1.1, Spark SQL, Spark MLlib, Apache Hive,
REM    Hadoop HDFS, Scala 2.13, Python 3.12 (scipy, matplotlib)
REM ============================================================

echo ======================================================================
echo   NYC TAXI FAIRNESS AUDIT – FULL PIPELINE
echo   Spark 4.1.1  ^|  Scala 2.13  ^|  MLlib  ^|  Hive  ^|  Python 3.12
echo ======================================================================

REM ─── Environment ───
set SPARK_HOME=N:\CAI\sem 6\BGA\prj\spark-4.1.1-bin-hadoop3\spark-4.1.1-bin-hadoop3
set JAVA_HOME=C:\Program Files\Java\jdk-21
set HADOOP_HOME=%SPARK_HOME%
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%
set PYSPARK_PYTHON=N:\CAI\sem 6\BGA\prj\venv\Scripts\python.exe
set PYTHON=N:\CAI\sem 6\BGA\prj\venv\Scripts\python.exe

cd /d "N:\CAI\sem 6\BGA\prj"

REM ─── Stage 1 : Data Cleaning (Scala / Spark) ───
echo.
echo [STAGE 1/5] DATA CLEANING  (Scala / Spark SQL)
echo   Cleaning 47M+ raw taxi records...
echo ----------------------------------------------------------------------
call "%SPARK_HOME%\bin\spark-shell.cmd" --driver-memory 8g ^
     -i scripts/scala/01_DataCleaning.scala
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] Stage 1 failed! & pause & exit /b 1 )

REM ─── Stage 2 : Data Enrichment (Scala / Spark) ───
echo.
echo [STAGE 2/5] DATA ENRICHMENT  (Scala / Spark SQL + Broadcast Join)
echo   Joining taxi data with census income by GPS zone mapping...
echo ----------------------------------------------------------------------
call "%SPARK_HOME%\bin\spark-shell.cmd" --driver-memory 8g ^
     -i scripts/scala/02_DataEnrichment.scala
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] Stage 2 failed! & pause & exit /b 1 )

REM ─── Stage 3 : ML Pipeline + Bias Detection (Scala / Spark MLlib) ───
echo.
echo [STAGE 3/5] ML PIPELINE + BIAS ANALYSIS  (Spark MLlib Random Forest)
echo   Training Baseline (with income) + Fair (without income) models...
echo   Computing Demographic Parity, Equalized Odds, Financial Impact...
echo ----------------------------------------------------------------------
call "%SPARK_HOME%\bin\spark-shell.cmd" --driver-memory 8g ^
     -i scripts/scala/03_MLPipeline.scala
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] Stage 3 failed! & pause & exit /b 1 )

REM ─── Stage 4 : Statistical Bias Analysis (Python / scipy) ───
echo.
echo [STAGE 4/5] STATISTICAL ANALYSIS  (Python – scipy, numpy, pandas)
echo   T-tests, Cohen's d, individual fairness, comprehensive report...
echo ----------------------------------------------------------------------
call "%PYTHON%" scripts/bias_analysis/01_bias_detection.py
if %ERRORLEVEL% NEQ 0 ( echo [WARN] Bias detection returned non-zero )

call "%PYTHON%" scripts/bias_analysis/02_fairness_metrics.py
if %ERRORLEVEL% NEQ 0 ( echo [WARN] Fairness metrics returned non-zero )

REM ─── Stage 5 : Visualizations (Python / matplotlib + seaborn) ───
echo.
echo [STAGE 5/5] VISUALIZATIONS  (Python – matplotlib, seaborn)
echo   Generating 6 publication-quality charts + Tableau export...
echo ----------------------------------------------------------------------
call "%PYTHON%" scripts/visualizations/generate_all_plots.py
if %ERRORLEVEL% NEQ 0 ( echo [WARN] Visualization generation returned non-zero )

REM ─── Done ───
echo.
echo ======================================================================
echo   PIPELINE COMPLETE
echo ======================================================================
echo.
echo   Frameworks Used:
echo     - Apache Spark 4.1.1  (Spark SQL, Spark ML Pipeline, MLlib)
echo     - Apache Hive 4.0.1   (DDL in hive/create_tables.hql)
echo     - Apache Kafka 7.6    (streaming simulation)
echo     - Hadoop HDFS 3.2.1   (distributed storage, bde2020)
echo     - Scala 2.13          (all Spark scripts)
echo     - Python 3.12         (scipy stats, matplotlib viz, kafka)
echo.
echo   Output Artifacts:
echo     output/processed/taxi_cleaned/     cleaned Parquet (Snappy)
echo     output/processed/taxi_enriched/    enriched with census income
echo     output/models/baseline_model/      MLlib RF (with income)
echo     output/models/fair_model/          MLlib RF (without income)
echo     output/results/predictions/        all predictions Parquet
echo     output/results/bias_analysis/      CSV + JSON reports
echo     output/results/fairness_metrics/   JSON report
echo     output/visualizations/             6 PNG charts + Tableau CSV
echo.
echo   Docker Services (docker compose up -d):
echo     HDFS     : namenode(9870) datanode(9864) YARN(8088)
echo     Hive     : metastore(9083) hiveserver2(10000,10002)
echo     Kafka    : zookeeper(2181) broker(9092)
echo ======================================================================
pause
