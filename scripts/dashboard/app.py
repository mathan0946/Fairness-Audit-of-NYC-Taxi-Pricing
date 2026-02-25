#!/usr/bin/env python3
"""
NYC Taxi Fairness Audit – Real-Time Web Dashboard
===================================================
Flask dashboard with Kafka streaming, Hive schema browser, and bias
analysis results.

Features:
  - Real-time Kafka consumer stats (SSE push)
  - Producer start / stop / reset controls
  - Bias analysis & fairness metrics visualisation
  - Hive table-schema browser (parsed from create_tables.hql)
  - Interactive Chart.js charts
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
import re
import sys
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, Response, request

# ── Kafka (optional – dashboard works without it) ──────────────
try:
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
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
HQL_PATH = os.path.join(BASE_DIR, "hive", "create_tables.hql")

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
    "consumer_running": False,
    "recent_trips": [],
    "borough_counts": {},
    "fare_sum": 0.0,
    "distance_sum": 0.0,
}
kafka_lock = threading.Lock()
consumer_thread = None
stop_consumer = threading.Event()

# Producer state
producer_thread = None
producer_lock = threading.Lock()
stop_producer = threading.Event()
producer_stats = {
    "running": False,
    "sent": 0,
    "rate": 0,
    "errors": 0,
    "target_limit": 0,
    "last_error": "",
}


# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_json(path):
    full = os.path.join(RESULTS_DIR, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv_rows(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def load_spark_csv(directory):
    rows = []
    if not os.path.isdir(directory):
        return rows
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


# ════════════════════════════════════════════════════════════════
#  HIVE SCHEMA PARSER
# ════════════════════════════════════════════════════════════════

def parse_hive_schema():
    """Parse create_tables.hql line-by-line to extract table / view defs."""
    if not os.path.exists(HQL_PATH):
        return {"database": "taxi_fairness", "tables": [], "views": [],
                "total_columns": 0, "raw": ""}

    with open(HQL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    content = "".join(lines)

    # ── Tables (line-by-line to avoid paren-in-COMMENT issues) ──
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"CREATE\s+EXTERNAL\s+TABLE\s+(\w+)\s*\(", line,
                     re.IGNORECASE)
        if m:
            table_name = m.group(1)
            columns, storage, location = [], "TEXTFILE", ""
            i += 1
            # Read columns until closing ')'
            while i < len(lines):
                cline = lines[i].strip()
                if cline.startswith(")"):
                    break
                if cline and not cline.startswith("--"):
                    cline_clean = cline.rstrip(",")
                    col_m = re.match(r"(\w+)\s+([\w]+)", cline_clean)
                    if col_m:
                        comment = ""
                        cm = re.search(r"COMMENT\s+['\"](.+?)['\"]",
                                       cline_clean)
                        if cm:
                            comment = cm.group(1)
                        columns.append({
                            "name": col_m.group(1),
                            "type": col_m.group(2).upper(),
                            "comment": comment,
                        })
                i += 1
            # Read storage / location until ';'
            while i < len(lines):
                pline = lines[i].strip()
                if "STORED AS PARQUET" in pline.upper():
                    storage = "PARQUET"
                elif "STORED AS ORC" in pline.upper():
                    storage = "ORC"
                lm = re.search(r"LOCATION\s+'([^']+)'", pline)
                if lm:
                    location = lm.group(1)
                if pline.endswith(";"):
                    break
                i += 1
            tables.append({
                "name": table_name,
                "columns": columns,
                "storage": storage,
                "location": location,
            })
        i += 1

    # ── Views ──
    views = []
    view_re = re.compile(
        r"CREATE\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+AS\s+(.*?);",
        re.DOTALL | re.IGNORECASE,
    )
    for m in view_re.finditer(content):
        views.append({"name": m.group(1), "query": m.group(2).strip()})

    return {
        "database": "taxi_fairness",
        "tables": tables,
        "views": views,
        "total_columns": sum(len(t["columns"]) for t in tables),
        "raw": content,
    }


def get_all_data():
    """Load all pipeline results + Hive schema."""
    bias_report = load_json(os.path.join("bias_analysis", "bias_report.json"))
    fairness_report = load_json(os.path.join("fairness_metrics",
                                             "fairness_metrics_report.json"))
    bias_summary = (
        load_spark_csv(os.path.join(RESULTS_DIR, "bias_analysis",
                                    "bias_summary_csv"))
        or load_csv_rows(os.path.join(RESULTS_DIR, "bias_analysis",
                                      "bias_summary_pandas.csv"))
    )
    borough_bias = load_spark_csv(
        os.path.join(RESULTS_DIR, "bias_analysis", "borough_bias_csv"))
    tableau_data = load_csv_rows(os.path.join(VIZ_DIR, "tableau_data.csv"))

    charts = []
    if os.path.isdir(VIZ_DIR):
        charts = sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".png"))

    return {
        "bias_report": bias_report,
        "fairness_report": fairness_report,
        "bias_summary": bias_summary,
        "borough_bias": borough_bias,
        "tableau_data": tableau_data,
        "charts": charts,
        "hive_schema": parse_hive_schema(),
    }


# ════════════════════════════════════════════════════════════════
#  KAFKA BACKGROUND CONSUMER
# ════════════════════════════════════════════════════════════════

# Borough bounding boxes — **ordered** from most geographically
# isolated to broadest so that the first match wins and overlaps
# are minimised.  Manhattan's east boundary is tightened to the
# East River (~-73.934) to avoid pulling Queens / Brooklyn trips.
# NOTE: Yellow-cab data is inherently Manhattan-heavy (~65 %
#       of all pickups).  That dominance is real, not a bug.
BOROUGH_BOUNDS = [
    # (name,           lat_min, lat_max,  lon_min,   lon_max)
    ("Staten Island",  40.496,  40.651,  -74.255,   -74.052),
    ("Bronx",          40.800,  40.917,  -73.933,   -73.748),
    ("Manhattan",      40.700,  40.882,  -74.019,   -73.934),
    ("Brooklyn",       40.570,  40.739,  -74.042,   -73.860),
    ("Queens",         40.530,  40.812,  -73.935,   -73.700),
]


def get_borough(lat, lon):
    """Bounding-box borough lookup (priority-ordered)."""
    if lat is None or lon is None:
        return "Unknown"
    for name, lat_min, lat_max, lon_min, lon_max in BOROUGH_BOUNDS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Unknown"


def kafka_consumer_loop():
    """Background thread: consume messages and update live stats.
    Auto-retries connection every 5 s until Kafka is available."""
    global kafka_stats
    if not KAFKA_AVAILABLE:
        return

    consumer = None
    while not stop_consumer.is_set():
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
            break  # connected – exit retry loop
        except Exception:
            with kafka_lock:
                kafka_stats["connected"] = False
            for _ in range(5):          # wait 5 s in 1-s increments
                if stop_consumer.is_set():
                    return
                time.sleep(1)

    if stop_consumer.is_set() or consumer is None:
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

                        lat = trip.get("pickup_latitude")
                        lon = trip.get("pickup_longitude")
                        borough = get_borough(
                            float(lat) if lat else None,
                            float(lon) if lon else None,
                        )
                        kafka_stats["borough_counts"][borough] = (
                            kafka_stats["borough_counts"].get(borough, 0) + 1
                        )

                        fare = trip.get("fare_amount")
                        dist = trip.get("trip_distance")
                        if fare:
                            kafka_stats["fare_sum"] += float(fare)
                        if dist:
                            kafka_stats["distance_sum"] += float(dist)

                        kafka_stats["recent_trips"].append({
                            "pickup": trip.get("tpep_pickup_datetime", ""),
                            "fare": fare,
                            "distance": dist,
                            "passengers": trip.get("passenger_count"),
                            "borough": borough,
                        })
                        if len(kafka_stats["recent_trips"]) > 20:
                            kafka_stats["recent_trips"] = \
                                kafka_stats["recent_trips"][-20:]

                    count_window += 1

            now = time.time()
            if now - window_start >= 1.0:
                with kafka_lock:
                    kafka_stats["messages_per_sec"] = (
                        count_window / (now - window_start)
                    )
                count_window = 0
                window_start = now
        except Exception:
            time.sleep(1)

    consumer.close()
    with kafka_lock:
        kafka_stats["consumer_running"] = False


def get_topic_info():
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
                end = consumer.end_offsets(tps)
                begin = consumer.beginning_offsets(tps)
                for tp in tps:
                    total_messages += end[tp] - begin[tp]
        consumer.close()
        return {"available": True, "connected": True,
                "topic_exists": topic_exists,
                "total_messages": total_messages,
                "topics": list(topics)}
    except Exception as e:
        return {"available": True, "connected": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
#  KAFKA PRODUCER (controlled from dashboard)
# ════════════════════════════════════════════════════════════════

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


def check_kafka_connectivity(timeout_ms=4000):
    """Quick check: can we reach the Kafka broker?"""
    if not KAFKA_AVAILABLE:
        return False, "kafka-python not installed"
    import socket
    host_port = KAFKA_BOOTSTRAP.split(",")[0].strip()
    host, _, port = host_port.partition(":")
    port = int(port) if port else 9092
    try:
        sock = socket.create_connection((host, port), timeout=timeout_ms / 1000)
        sock.close()
        return True, ""
    except Exception as e:
        return False, f"Cannot reach Kafka at {host}:{port} – {e}"


def kafka_producer_loop(csv_path, rate, limit):
    """Background thread: stream CSV rows to Kafka."""
    if not KAFKA_AVAILABLE:
        with producer_lock:
            producer_stats["running"] = False
            producer_stats["last_error"] = "kafka-python not installed"
        return

    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all", retries=3, batch_size=16384, linger_ms=10,
            request_timeout_ms=5000, max_block_ms=5000,
        )
    except Exception as e:
        with producer_lock:
            producer_stats["running"] = False
            producer_stats["last_error"] = f"Kafka connection failed: {e}"
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
                with producer_lock:
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
                parsed["ingestion_timestamp"] = (
                    datetime.now(tz=timezone.utc).isoformat()
                )

                try:
                    producer.send(KAFKA_TOPIC, value=parsed)
                    with producer_lock:
                        producer_stats["sent"] += 1
                except Exception:
                    with producer_lock:
                        producer_stats["errors"] += 1

                elapsed = time.time() - start
                with producer_lock:
                    producer_stats["rate"] = (
                        int(producer_stats["sent"] / elapsed) if elapsed > 0
                        else 0
                    )

                if delay > 0:
                    time.sleep(delay)
    except Exception:
        pass

    producer.flush()
    producer.close()
    with producer_lock:
        producer_stats["running"] = False


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html",
                           data=get_all_data(),
                           kafka_available=KAFKA_AVAILABLE)


@app.route("/api/results")
def api_results():
    return jsonify(get_all_data())


@app.route("/api/hive/schema")
def api_hive_schema():
    return jsonify(parse_hive_schema())


@app.route("/api/kafka/stats")
def api_kafka_stats():
    with kafka_lock:
        stats = dict(kafka_stats)
        stats["recent_trips"] = list(stats["recent_trips"])
        stats["borough_counts"] = dict(stats["borough_counts"])
    with producer_lock:
        stats["producer"] = dict(producer_stats)
    return jsonify(stats)


@app.route("/api/kafka/info")
def api_kafka_info():
    return jsonify(get_topic_info())


@app.route("/api/kafka/start-producer", methods=["POST"])
def api_start_producer():
    """Start the Kafka producer."""
    global producer_thread

    if not KAFKA_AVAILABLE:
        return jsonify({"error": "Kafka not available"}), 400

    # Wait for any previous thread to finish
    if producer_thread and producer_thread.is_alive():
        stop_producer.set()
        producer_thread.join(timeout=5)
        if producer_thread.is_alive():
            return jsonify({"error": "Previous producer still shutting down"}), 409

    with producer_lock:
        if producer_stats["running"]:
            return jsonify({"error": "Producer already running"}), 400

    body = request.json or {}
    rate = int(body.get("rate", 500))
    limit = int(body.get("limit", 5000))
    csv_file = body.get("csv", "data/yellow_tripdata_2016-01.csv")
    csv_path = (os.path.join(BASE_DIR, csv_file)
                if not os.path.isabs(csv_file) else csv_file)

    if not os.path.exists(csv_path):
        return jsonify({"error": f"CSV not found: {csv_path}"}), 404

    # Pre-flight: check Kafka is reachable
    reachable, reason = check_kafka_connectivity(timeout_ms=3000)
    if not reachable:
        return jsonify({"error": f"Kafka broker not reachable. {reason}"}), 503

    # Reset producer state
    stop_producer.clear()
    with producer_lock:
        producer_stats["running"] = True
        producer_stats["sent"] = 0
        producer_stats["errors"] = 0
        producer_stats["rate"] = 0
        producer_stats["target_limit"] = limit
        producer_stats["last_error"] = ""

    producer_thread = threading.Thread(
        target=kafka_producer_loop, args=(csv_path, rate, limit), daemon=True,
    )
    producer_thread.start()
    return jsonify({"status": "started", "rate": rate, "limit": limit})


@app.route("/api/kafka/stop-producer", methods=["POST"])
def api_stop_producer():
    """Signal the Kafka producer to stop."""
    global producer_thread
    stop_producer.set()
    if producer_thread and producer_thread.is_alive():
        producer_thread.join(timeout=3)
    return jsonify({"status": "stopped"})


@app.route("/api/kafka/reset-stats", methods=["POST"])
def api_reset_stats():
    """Clear all live Kafka counters."""
    with kafka_lock:
        kafka_stats["messages_consumed"] = 0
        kafka_stats["messages_per_sec"] = 0.0
        kafka_stats["borough_counts"] = {}
        kafka_stats["fare_sum"] = 0.0
        kafka_stats["distance_sum"] = 0.0
        kafka_stats["recent_trips"] = []
    return jsonify({"status": "reset"})


@app.route("/api/kafka/stream")
def api_kafka_stream():
    """SSE endpoint – pushes stats every second."""
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
                }
            with producer_lock:
                stats["producer"] = dict(producer_stats)
            yield f"data: {json.dumps(stats)}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/charts/<filename>")
def serve_chart(filename):
    from flask import send_from_directory
    return send_from_directory(VIZ_DIR, filename)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    global consumer_thread
    parser = argparse.ArgumentParser(description="NYC Taxi Fairness Dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-kafka", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("  NYC Taxi Fairness Audit – Web Dashboard")
    print("=" * 62)
    print(f"  URL   : http://{args.host}:{args.port}")
    print(f"  Kafka : {'enabled' if KAFKA_AVAILABLE and not args.no_kafka else 'disabled'}")
    print(f"  Hive  : schema from {HQL_PATH}")
    print()

    if KAFKA_AVAILABLE and not args.no_kafka:
        consumer_thread = threading.Thread(target=kafka_consumer_loop,
                                           daemon=True)
        consumer_thread.start()
        print("  Kafka consumer started (topic: taxi-trips-raw)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
