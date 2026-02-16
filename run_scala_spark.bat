@echo off
REM ============================================
REM NYC Taxi Fairness Audit - Scala Spark Runner
REM ============================================

echo.
echo ========================================
echo NYC Taxi Fairness Audit (Scala)
echo ========================================
echo.

REM Check if Spark is available
where spark-submit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: spark-submit not found in PATH
    echo.
    echo Please ensure Apache Spark is installed and SPARK_HOME is set.
    echo Example: set SPARK_HOME=C:\spark-3.5.1
    echo          set PATH=%%PATH%%;%%SPARK_HOME%%\bin
    pause
    exit /b 1
)

REM Set project directory
set PROJECT_DIR=%~dp0

REM Run Spark with Scala script
echo Running Scala Spark analysis...
echo.

spark-shell ^
    --master local[*] ^
    --driver-memory 4g ^
    --conf spark.sql.shuffle.partitions=200 ^
    -i "%PROJECT_DIR%scripts\scala\DataAnalysis.scala"

echo.
echo ========================================
echo Script execution complete!
echo ========================================
pause
