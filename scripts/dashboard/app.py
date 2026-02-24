#!/usr/bin/env python3
"""
NYC Taxi Fairness Audit – Real-Time Web Dashboard
===================================================
Flask web dashboard with live Kafka streaming integration.

Features:
  - Real-time Kafka consumer stats (SSE)
  - Bias analysis results & fairness metrics
  - Interactive Chart.js visualizations
  - Borough-level and income-level breakdown
  - Kafka producer control (start/stop streaming)
  - Pipeline architecture overview

Usage:
    python scripts/dashboard/app.py [--port 5050]

Requires:
    pip install flask kafka-python-ng
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, Response, request

# ── Kafka (optional – dashboard works without it) ──────────────
try:
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
    from kafka.admin import KafkaAdminClient
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
VIZ_DIR = os.path.join(BASE_DIR, "output", "visualizations")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = "taxi-trips-raw"

# ── Flask App ──────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ── Global State ───────────────────────────────────────────────
kafka_stats = {
    "connected": False,
    "messages_consumed": 0,
    "messages_per_sec": 0.0,
    "last_message": None,
    "start_time": None,
    "batches": 0,
    "topic_exists": False,
    "total_topic_messages": 0,
    "consumer_running": False,
    "recent_trips": [],       # last 20 trips for live feed
    "borough_counts": {},     # real-time borough distribution
    "fare_sum": 0.0,
    "distance_sum": 0.0,
}
kafka_lock = threading.Lock()
consumer_thread = None
stop_consumer = threading.Event()

# Producer control
producer_thread = None
stop_producer = threading.Event()
producer_stats = {
    "running": False,
    "sent": 0,
    "rate": 0,
    "errors": 0,
}


# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_json(filename):
    """Load a JSON file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv_rows(filepath):
    """Load CSV file as list of dicts."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    return []


def load_spark_csv(directory):
    """Load Spark-partitioned CSV (part-* files) from a directory."""
    rows = []
    if not os.path.isdir(directory):
        return rows
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows.extend(reader)
    return rows


def get_all_data():
    """Load all pipeline results into a single dict."""
    bias_report = load_json(os.path.join("bias_analysis", "bias_report.json"))
    fairness_report = load_json(os.path.join("fairness_metrics", "fairness_metrics_report.json"))

    bias_summary = load_spark_csv(
        os.path.join(RESULTS_DIR, "bias_analysis", "bias_summary_csv")
    )
    if not bias_summary:
        bias_summary = load_csv_rows(
            os.path.join(RESULTS_DIR, "bias_analysis", "bias_summary_pandas.csv")
        )

    borough_bias = load_spark_csv(
        os.path.join(RESULTS_DIR, "bias_analysis", "borough_bias_csv")
    )

    tableau_data = load_csv_rows(os.path.join(VIZ_DIR, "tableau_data.csv"))

    # Chart images available
    charts = []
    if os.path.isdir(VIZ_DIR):
        charts = sorted([f for f in os.listdir(VIZ_DIR) if f.endswith(".png")])

    return {
        "bias_report": bias_report,
        "fairness_report": fairness_report,
        "bias_summary": bias_summary,
        "borough_bias": borough_bias,
        "tableau_data": tableau_data,
        "charts": charts,
    }


# ════════════════════════════════════════════════════════════════
#  KAFKA BACKGROUND CONSUMER
# ════════════════════════════════════════════════════════════════

# NYC borough bounding boxes (simplified)
BOROUGH_BOUNDS = {
    "Manhattan":     (40.700, 40.880, -74.020, -73.907),
    "Brooklyn":      (40.570, 40.739, -74.042, -73.855),
    "Queens":        (40.541, 40.812, -73.962, -73.700),
    "Bronx":         (40.785, 40.917, -73.933, -73.748),
    "Staten Island": (40.496, 40.651, -74.255, -74.052),
}


def get_borough(lat, lon):
    """Simple bounding-box borough lookup."""
    if lat is None or lon is None:
        return "Unknown"
    for name, (lat_min, lat_max, lon_min, lon_max) in BOROUGH_BOUNDS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Unknown"


def kafka_consumer_loop():
    """Background thread: consume from Kafka and update stats."""
    global kafka_stats
    if not KAFKA_AVAILABLE:
        return

    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id="dashboard-consumer",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=2000,
            max_poll_records=100,
        )
    except NoBrokersAvailable:
        with kafka_lock:
            kafka_stats["connected"] = False
        return
    except Exception:
        with kafka_lock:
            kafka_stats["connected"] = False
        return

    with kafka_lock:
        kafka_stats["connected"] = True
        kafka_stats["consumer_running"] = True
        kafka_stats["start_time"] = time.time()

    count_window = 0
    window_start = time.time()

    while not stop_consumer.is_set():
        try:
            records = consumer.poll(timeout_ms=1000, max_records=200)
            for tp, messages in records.items():
                for msg in messages:
                    trip = msg.value
                    with kafka_lock:
                        kafka_stats["messages_consumed"] += 1
                        kafka_stats["last_message"] = trip

                        # Track borough
                        lat = trip.get("pickup_latitude")
                        lon = trip.get("pickup_longitude")
                        borough = get_borough(
                            float(lat) if lat else None,
                            float(lon) if lon else None
                        )
                        kafka_stats["borough_counts"][borough] = \
                            kafka_stats["borough_counts"].get(borough, 0) + 1

                        # Track fare/distance
                        fare = trip.get("fare_amount")
                        dist = trip.get("trip_distance")
                        if fare:
                            kafka_stats["fare_sum"] += float(fare)
                        if dist:
                            kafka_stats["distance_sum"] += float(dist)

                        # Recent trips (keep last 20)
                        kafka_stats["recent_trips"].append({
                            "pickup": trip.get("tpep_pickup_datetime", ""),
                            "fare": fare,
                            "distance": dist,
                            "passengers": trip.get("passenger_count"),
                            "borough": borough,
                        })
                        if len(kafka_stats["recent_trips"]) > 20:
                            kafka_stats["recent_trips"] = kafka_stats["recent_trips"][-20:]

                    count_window += 1

            # Update rate every second
            now = time.time()
            if now - window_start >= 1.0:
                with kafka_lock:
                    kafka_stats["messages_per_sec"] = count_window / (now - window_start)
                count_window = 0
                window_start = now

        except Exception:
            time.sleep(1)

    consumer.close()
    with kafka_lock:
        kafka_stats["consumer_running"] = False


def get_topic_info():
    """Get Kafka topic metadata."""
    if not KAFKA_AVAILABLE:
        return {"available": False}
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            consumer_timeout_ms=3000,
        )
        topics = consumer.topics()
        topic_exists = KAFKA_TOPIC in topics

        total_messages = 0
        if topic_exists:
            partitions = consumer.partitions_for_topic(KAFKA_TOPIC)
            if partitions:
                tps = [TopicPartition(KAFKA_TOPIC, p) for p in partitions]
                consumer.assign(tps)
                end_offsets = consumer.end_offsets(tps)
                begin_offsets = consumer.beginning_offsets(tps)
                for tp in tps:
                    total_messages += end_offsets[tp] - begin_offsets[tp]

        consumer.close()
        return {
            "available": True,
            "connected": True,
            "topic_exists": topic_exists,
            "total_messages": total_messages,
            "topics": list(topics),
        }
    except Exception as e:
        return {"available": True, "connected": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
#  KAFKA PRODUCER (controlled from dashboard)
# ════════════════════════════════════════════════════════════════

def kafka_producer_loop(csv_path, rate, limit):
    """Background thread: stream CSV to Kafka."""
    global producer_stats
    if not KAFKA_AVAILABLE:
        return

    producer_stats["running"] = True
    producer_stats["sent"] = 0
    producer_stats["errors"] = 0

    CSV_COLUMNS = [
        "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance", "pickup_longitude", "pickup_latitude",
        "RatecodeID", "store_and_fwd_flag", "dropoff_longitude", "dropoff_latitude",
        "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount",
        "tolls_amount", "improvement_surcharge", "total_amount",
    ]
    NUMERIC = {"VendorID", "passenger_count", "RatecodeID", "payment_type"}
    FLOATS = {
        "trip_distance", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "fare_amount", "extra",
        "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
        "total_amount",
    }

    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all", retries=3, batch_size=16384, linger_ms=10,
        )
    except Exception:
        producer_stats["running"] = False
        return

    delay = 1.0 / rate if rate > 0 else 0
    start = time.time()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=CSV_COLUMNS)
            next(reader)  # skip header
            for row in reader:
                if stop_producer.is_set():
                    break
                if 0 < limit <= producer_stats["sent"]:
                    break

                parsed = {}
                for k, v in row.items():
                    if not v or not v.strip():
                        parsed[k] = None
                    elif k in NUMERIC:
                        try:
                            parsed[k] = int(float(v))
                        except ValueError:
                            parsed[k] = None
                    elif k in FLOATS:
                        try:
                            parsed[k] = round(float(v), 6)
                        except ValueError:
                            parsed[k] = None
                    else:
                        parsed[k] = v.strip()
                parsed["ingestion_timestamp"] = datetime.now(tz=timezone.utc).isoformat()

                try:
                    producer.send(KAFKA_TOPIC, value=parsed)
                    producer_stats["sent"] += 1
                except Exception:
                    producer_stats["errors"] += 1

                elapsed = time.time() - start
                producer_stats["rate"] = int(producer_stats["sent"] / elapsed) if elapsed > 0 else 0

                if delay > 0:
                    time.sleep(delay)
    except Exception:
        pass

    producer.flush()
    producer.close()
    producer_stats["running"] = False


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Main dashboard page."""
    data = get_all_data()
    return render_template("dashboard.html", data=data, kafka_available=KAFKA_AVAILABLE)


@app.route("/api/results")
def api_results():
    """Return all pipeline results as JSON."""
    return jsonify(get_all_data())


@app.route("/api/kafka/stats")
def api_kafka_stats():
    """Return current Kafka consumer stats."""
    with kafka_lock:
        stats = dict(kafka_stats)
        stats["recent_trips"] = list(stats["recent_trips"])
        stats["borough_counts"] = dict(stats["borough_counts"])
    stats["producer"] = dict(producer_stats)
    return jsonify(stats)


@app.route("/api/kafka/info")
def api_kafka_info():
    """Return Kafka topic metadata."""
    return jsonify(get_topic_info())


@app.route("/api/kafka/start-producer", methods=["POST"])
def api_start_producer():
    """Start the Kafka producer in background."""
    global producer_thread
    if not KAFKA_AVAILABLE:
        return jsonify({"error": "Kafka not available"}), 400
    if producer_stats["running"]:
        return jsonify({"error": "Producer already running"}), 400

    body = request.json or {}
    rate = body.get("rate", 500)
    limit = body.get("limit", 5000)
    csv_file = body.get("csv", "data/yellow_tripdata_2016-01.csv")
    csv_path = os.path.join(BASE_DIR, csv_file) if not os.path.isabs(csv_file) else csv_file

    if not os.path.exists(csv_path):
        return jsonify({"error": f"CSV not found: {csv_path}"}), 404

    stop_producer.clear()
    producer_thread = threading.Thread(
        target=kafka_producer_loop, args=(csv_path, rate, limit), daemon=True
    )
    producer_thread.start()
    return jsonify({"status": "started", "rate": rate, "limit": limit})


@app.route("/api/kafka/stop-producer", methods=["POST"])
def api_stop_producer():
    """Stop the Kafka producer."""
    stop_producer.set()
    return jsonify({"status": "stopping"})


@app.route("/api/kafka/stream")
def api_kafka_stream():
    """Server-Sent Events stream for real-time Kafka stats."""
    def generate():
        while True:
            with kafka_lock:
                stats = {
                    "messages_consumed": kafka_stats["messages_consumed"],
                    "messages_per_sec": round(kafka_stats["messages_per_sec"], 1),
                    "connected": kafka_stats["connected"],
                    "consumer_running": kafka_stats["consumer_running"],
                    "borough_counts": dict(kafka_stats["borough_counts"]),
                    "fare_sum": round(kafka_stats["fare_sum"], 2),
                    "distance_sum": round(kafka_stats["distance_sum"], 2),
                    "recent_trips": list(kafka_stats["recent_trips"][-5:]),
                    "producer": dict(producer_stats),
                }
            yield f"data: {json.dumps(stats)}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/charts/<filename>")
def serve_chart(filename):
    """Serve a visualization PNG."""
    from flask import send_from_directory
    return send_from_directory(VIZ_DIR, filename)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    global consumer_thread

    parser = argparse.ArgumentParser(description="NYC Taxi Fairness Dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Port (default 5050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--no-kafka", action="store_true", help="Disable Kafka integration")
    args = parser.parse_args()

    print("=" * 62)
    print("  NYC Taxi Fairness Audit – Web Dashboard")
    print("=" * 62)
    print(f"  URL      : http://{args.host}:{args.port}")
    print(f"  Kafka    : {'enabled' if KAFKA_AVAILABLE and not args.no_kafka else 'disabled'}")
    print(f"  Results  : {RESULTS_DIR}")
    print()

    # Start Kafka consumer thread
    if KAFKA_AVAILABLE and not args.no_kafka:
        consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
        consumer_thread.start()
        print("  Kafka consumer thread started (topic: taxi-trips-raw)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
