"""
============================================
NYC Taxi Fairness Audit - Run All Pipeline
============================================
Master script to run the entire pipeline locally

Usage: python run_pipeline.py [--skip-spark]

Author: Big Data Analytics Project
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"⚠️ Warning: Script returned non-zero exit code")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       NYC TAXI FAIRNESS AUDIT - PIPELINE RUNNER              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  This script runs the complete analysis pipeline locally.    ║
    ║  For full Big Data processing, use Spark cluster commands.   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    skip_spark = '--skip-spark' in sys.argv
    
    if skip_spark:
        print("⚠️  Skipping Spark scripts (--skip-spark flag detected)")
    
    # Create output directories
    output_dirs = [
        'output/processed/taxi_cleaned',
        'output/processed/taxi_enriched',
        'output/processed/ml_ready',
        'output/models/baseline_model',
        'output/models/fair_model',
        'output/results/baseline_predictions',
        'output/results/fair_predictions',
        'output/results/bias_analysis',
        'output/results/fairness_metrics',
        'output/visualizations'
    ]
    
    print("\n📁 Creating output directories...")
    for dir_path in output_dirs:
        os.makedirs(PROJECT_ROOT / dir_path, exist_ok=True)
        print(f"   ✓ {dir_path}")
    
    # Pipeline steps
    pipeline = []
    
    if not skip_spark:
        pipeline.extend([
            ('scripts/data_processing/01_data_cleaning.py', 
             'Stage 1: Data Cleaning (PySpark)'),
            ('scripts/data_processing/02_data_enrichment.py', 
             'Stage 2: Data Enrichment (PySpark)'),
            ('scripts/feature_engineering/feature_engineering.py', 
             'Stage 3: Feature Engineering (PySpark)'),
            ('scripts/ml_models/01_baseline_model.py', 
             'Stage 4a: Baseline Model Training (Spark MLlib)'),
            ('scripts/ml_models/02_fair_model.py', 
             'Stage 4b: Fair Model Training (Spark MLlib)'),
        ])
    
    pipeline.extend([
        ('scripts/bias_analysis/01_bias_detection.py', 
         'Stage 5: Bias Detection Analysis'),
        ('scripts/bias_analysis/02_fairness_metrics.py', 
         'Stage 6: Fairness Metrics Calculation'),
        ('scripts/visualizations/generate_all_plots.py', 
         'Stage 7: Visualization Generation'),
    ])
    
    # Run pipeline
    results = {}
    for script, description in pipeline:
        script_path = PROJECT_ROOT / script
        if script_path.exists():
            success = run_script(script_path, description)
            results[description] = '✅' if success else '❌'
        else:
            print(f"\n⚠️ Script not found: {script}")
            results[description] = '⚠️ Not found'
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    for step, status in results.items():
        print(f"  {status} {step}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"""
    📊 Output locations:
       - Processed data:  output/processed/
       - ML models:       output/models/
       - Analysis:        output/results/
       - Visualizations:  output/visualizations/
    
    📋 Next steps:
       1. Review visualizations in output/visualizations/
       2. Check bias_report.json in output/results/bias_analysis/
       3. Open notebooks/fairness_audit_analysis.ipynb for interactive analysis
       4. Create Tableau dashboard using output/visualizations/tableau_data.csv
    """)


if __name__ == "__main__":
    main()
