#!/usr/bin/env python3
"""
Kafka Consumer – Reads streamed taxi data and writes to HDFS / local
=====================================================================
Consumes JSON messages from the taxi-trips-raw Kafka topic and writes
them as newline-delimited JSON (suitable for Hive/Spark ingestion).

Usage:
    python scripts/kafka/kafka_consumer.py [--output output/kafka_ingest] [--batch-size 1000]

This consumer:
  1. Reads from 'taxi-trips-raw' topic
  2. Batches messages
  3. Writes as JSON-lines files (for Hive EXTERNAL TABLE or Spark)
  4. Prints real-time statistics
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    print("ERROR: kafka-python not installed. Run: pip install kafka-python-ng")
    sys.exit(1)


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "taxi-trips-raw"
GROUP_ID = "taxi-fairness-consumer"


def create_consumer(bootstrap: str, retries: int = 5) -> KafkaConsumer:
    """Create Kafka consumer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=bootstrap,
                group_id=GROUP_ID,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=30000,  # 30s timeout
                max_poll_records=500,
            )
            print(f"  Connected to Kafka at {bootstrap}")
            print(f"  Subscribed to topic: {TOPIC}")
            return consumer
        except NoBrokersAvailable:
            if attempt < retries:
                print(f"  Broker not available (attempt {attempt}/{retries}), retrying in 5s...")
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Could not connect to Kafka")


def consume_and_write(consumer: KafkaConsumer, output_dir: str, batch_size: int):
    """Consume messages and write to output files in batches."""
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    batch = []
    batch_num = 0
    start = time.time()

    print(f"  Output dir : {output_dir}")
    print(f"  Batch size : {batch_size}")
    print(f"  Waiting for messages...")
    print()

    try:
        for message in consumer:
            batch.append(message.value)
            total += 1

            if len(batch) >= batch_size:
                batch_num += 1
                flush_batch(batch, output_dir, batch_num)
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"    Batch {batch_num} written  |  Total: {total:,}  |  Rate: {rate:.0f} msgs/sec")
                batch = []

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    # Write remaining
    if batch:
        batch_num += 1
        flush_batch(batch, output_dir, batch_num)

    elapsed = time.time() - start
    print()
    print(f"  ── Consumer Summary ──")
    print(f"  Messages consumed : {total:,}")
    print(f"  Batches written   : {batch_num}")
    print(f"  Elapsed           : {elapsed:.1f}s")
    print(f"  Output directory  : {output_dir}")

    return total


def flush_batch(batch: list, output_dir: str, batch_num: int):
    """Write a batch of messages to a JSON-lines file."""
    timestamp = datetime.now(tz=__import__('datetime').timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"taxi_batch_{batch_num:04d}_{timestamp}.jsonl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in batch:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Kafka taxi trip consumer")
    parser.add_argument("--output", default="output/kafka_ingest",
                        help="Output directory for ingested data")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Messages per output file")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP,
                        help="Kafka bootstrap servers")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kafka Consumer – NYC Taxi Trip Ingestion")
    print("=" * 60)
    print()

    # Resolve output path
    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), output_dir)

    consumer = create_consumer(args.bootstrap)
    total = consume_and_write(consumer, output_dir, args.batch_size)
    consumer.close()

    if total == 0:
        print("  No messages consumed. Make sure the producer has sent data.")
    else:
        print(f"  Done. {total:,} messages written to {output_dir}")


if __name__ == "__main__":
    main()
