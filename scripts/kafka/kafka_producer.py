#!/usr/bin/env python3
"""
Kafka Producer – Simulates Real-Time NYC Taxi Trip Streaming
=============================================================
Reads taxi trip CSV data and publishes records to a Kafka topic
as JSON messages, simulating a real-time data ingestion pipeline.

Usage:
    python scripts/kafka/kafka_producer.py [--rate 100] [--limit 5000]

Topic: taxi-trips-raw
"""

import argparse
import csv
import json
import time
import sys
import os
from datetime import datetime

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    print("ERROR: kafka-python not installed. Run: pip install kafka-python-ng")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "taxi-trips-raw"

CSV_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "pickup_longitude", "pickup_latitude",
    "RatecodeID", "store_and_fwd_flag", "dropoff_longitude", "dropoff_latitude",
    "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "improvement_surcharge", "total_amount"
]

NUMERIC_COLS = {
    "VendorID", "passenger_count", "RatecodeID", "payment_type"
}
FLOAT_COLS = {
    "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount"
}


def parse_row(row: dict) -> dict:
    """Convert CSV string values to proper types."""
    parsed = {}
    for k, v in row.items():
        if not v or v.strip() == "":
            parsed[k] = None
        elif k in NUMERIC_COLS:
            try:
                parsed[k] = int(float(v))
            except ValueError:
                parsed[k] = None
        elif k in FLOAT_COLS:
            try:
                parsed[k] = round(float(v), 6)
            except ValueError:
                parsed[k] = None
        else:
            parsed[k] = v.strip()
    parsed["ingestion_timestamp"] = datetime.now(tz=__import__('datetime').timezone.utc).isoformat()
    return parsed


def create_producer(bootstrap: str, retries: int = 5) -> KafkaProducer:
    """Create Kafka producer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                batch_size=16384,
                linger_ms=10,
            )
            print(f"  Connected to Kafka at {bootstrap}")
            return producer
        except NoBrokersAvailable:
            if attempt < retries:
                print(f"  Broker not available (attempt {attempt}/{retries}), retrying in 5s...")
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Could not connect to Kafka")


def stream_csv(csv_path: str, producer: KafkaProducer, rate: int, limit: int):
    """Stream CSV rows to Kafka topic."""
    sent = 0
    errors = 0
    start = time.time()
    delay = 1.0 / rate if rate > 0 else 0

    print(f"  Streaming from: {csv_path}")
    print(f"  Rate: {rate} msgs/sec | Limit: {limit if limit > 0 else 'unlimited'}")
    print(f"  Topic: {TOPIC}")
    print()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=CSV_COLUMNS)
        next(reader)  # skip header

        for row in reader:
            if 0 < limit <= sent:
                break

            try:
                msg = parse_row(row)
                # Use pickup datetime as key for partitioning
                key = msg.get("tpep_pickup_datetime", str(sent))
                producer.send(TOPIC, key=key, value=msg)
                sent += 1

                if sent % 500 == 0:
                    elapsed = time.time() - start
                    actual_rate = sent / elapsed if elapsed > 0 else 0
                    print(f"    Sent: {sent:,} msgs  |  Rate: {actual_rate:.0f} msgs/sec")

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    ERROR sending msg {sent}: {e}")

    producer.flush()
    elapsed = time.time() - start
    print()
    print(f"  ── Streaming Complete ──")
    print(f"  Messages sent : {sent:,}")
    print(f"  Errors        : {errors}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print(f"  Avg rate      : {sent / elapsed:.0f} msgs/sec" if elapsed > 0 else "")


def main():
    parser = argparse.ArgumentParser(description="Kafka taxi trip producer")
    parser.add_argument("--csv", default="data/yellow_tripdata_2016-01.csv",
                        help="Path to taxi CSV file")
    parser.add_argument("--rate", type=int, default=500,
                        help="Messages per second (0 = max speed)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Max messages to send (0 = all)")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP,
                        help="Kafka bootstrap servers")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kafka Producer – NYC Taxi Trip Streaming Simulation")
    print("=" * 60)
    print()

    # Resolve CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), csv_path)

    if not os.path.exists(csv_path):
        print(f"  ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    producer = create_producer(args.bootstrap)
    stream_csv(csv_path, producer, args.rate, args.limit)
    producer.close()


if __name__ == "__main__":
    main()
