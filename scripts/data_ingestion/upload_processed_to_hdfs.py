#!/usr/bin/env python3
"""
Upload processed Parquet & result CSV files to HDFS via WebHDFS REST API.
Run from project root:  python scripts/data_ingestion/upload_processed_to_hdfs.py
"""
import os
import sys
import glob
import requests

WEBHDFS = "http://localhost:9870/webhdfs/v1"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Local dir  →  HDFS destination dir
UPLOADS = [
    (
        os.path.join(BASE_DIR, "output", "processed", "taxi_cleaned"),
        "/bigdata/processed/taxi_cleaned",
        "*.parquet",
    ),
    (
        os.path.join(BASE_DIR, "output", "processed", "taxi_enriched"),
        "/bigdata/processed/taxi_enriched",
        "*.parquet",
    ),
    (
        os.path.join(BASE_DIR, "output", "results", "predictions"),
        "/bigdata/results/predictions",
        "part-*.parquet",
    ),
    (
        os.path.join(BASE_DIR, "output", "results", "bias_analysis", "bias_summary_csv"),
        "/bigdata/results/bias_summary_csv",
        "part-*",
    ),
    (
        os.path.join(BASE_DIR, "output", "results", "bias_analysis", "borough_bias_csv"),
        "/bigdata/results/borough_bias_csv",
        "part-*",
    ),
]


def hdfs_mkdir(hdfs_path: str):
    r = requests.put(
        f"{WEBHDFS}{hdfs_path}",
        params={"op": "MKDIRS", "user.name": "root"},
    )
    r.raise_for_status()


def _fix_redirect(url: str) -> str:
    """Replace internal Docker hostname in WebHDFS redirect with localhost."""
    from urllib.parse import urlparse, urlunparse
    p = urlparse(url)
    # The datanode is mapped to localhost:9864 on the host
    return urlunparse(p._replace(netloc=f"localhost:{p.port or 9864}"))


def hdfs_put(local_path: str, hdfs_path: str):
    """Upload a single file via WebHDFS two-step PUT."""
    params = {"op": "CREATE", "overwrite": "true", "user.name": "root"}
    # Step 1 – get redirect URL (do NOT follow – internal hostname unusable from host)
    r1 = requests.put(
        f"{WEBHDFS}{hdfs_path}",
        params=params,
        allow_redirects=False,
    )
    if r1.status_code not in (307, 201):
        raise RuntimeError(f"WebHDFS redirect failed {r1.status_code}: {r1.text[:200]}")

    redirect_url = r1.headers.get("Location")
    if not redirect_url:
        return  # 201 direct – nothing more needed

    # Rewrite container hostname → localhost so Windows can reach it
    redirect_url = _fix_redirect(redirect_url)

    # Step 2 – stream the file
    with open(local_path, "rb") as fh:
        r2 = requests.put(redirect_url, data=fh)
    r2.raise_for_status()


def main():
    total_files = 0
    total_bytes = 0

    for local_dir, hdfs_dir, pattern in UPLOADS:
        if not os.path.isdir(local_dir):
            print(f"  [SKIP] local dir not found: {local_dir}")
            continue

        files = sorted(glob.glob(os.path.join(local_dir, pattern)))
        # Exclude Hadoop CRC / hidden files
        files = [f for f in files if not os.path.basename(f).startswith(".")]

        if not files:
            print(f"  [SKIP] no matching files in {local_dir}")
            continue

        print(f"\n→ {hdfs_dir}  ({len(files)} files)")
        hdfs_mkdir(hdfs_dir)

        for local_file in files:
            fname = os.path.basename(local_file)
            hdfs_file = f"{hdfs_dir}/{fname}"
            size = os.path.getsize(local_file)
            sys.stdout.write(f"   uploading {fname} ({size/1024/1024:.1f} MB) … ")
            sys.stdout.flush()
            hdfs_put(local_file, hdfs_file)
            total_files += 1
            total_bytes += size
            print("done")

    print(f"\n✓ Uploaded {total_files} files  ({total_bytes/1024/1024:.1f} MB total)")


if __name__ == "__main__":
    main()
