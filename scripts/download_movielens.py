#!/usr/bin/env python3
"""Download MovieLens locally and convert ratings to implicit interactions.

MovieLens is an optional recommendation benchmark, not dating-app evidence. The
download is gated behind an explicit terms acknowledgement and is never
committed by this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
TERMS_URL = "https://grouplens.org/datasets/movielens/"
ARCHIVE_MEMBER = "ml-latest-small/ratings.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_ratings(
    ratings_path: Path,
    output_path: Path,
    *,
    positive_threshold: float = 4.0,
) -> dict[str, object]:
    """Convert positive MovieLens ratings into this lab's interaction schema."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_written = 0
    users: set[str] = set()
    items: set[str] = set()

    with ratings_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        expected = {"userId", "movieId", "rating", "timestamp"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected MovieLens ratings schema: {reader.fieldnames}")

        writer = csv.DictWriter(
            destination,
            fieldnames=["decidermemberid", "othermemberid", "timestamp", "like"],
        )
        writer.writeheader()
        for row in reader:
            rows_read += 1
            if float(row["rating"]) < positive_threshold:
                continue
            user_id = f"ml-user-{row['userId']}"
            item_id = f"ml-movie-{row['movieId']}"
            timestamp = datetime.fromtimestamp(
                int(row["timestamp"]), tz=timezone.utc
            ).isoformat()
            writer.writerow(
                {
                    "decidermemberid": user_id,
                    "othermemberid": item_id,
                    "timestamp": timestamp,
                    "like": 1,
                }
            )
            users.add(user_id)
            items.add(item_id)
            rows_written += 1

    return {
        "ratings_read": rows_read,
        "positive_interactions": rows_written,
        "users": len(users),
        "items": len(items),
        "positive_threshold": positive_threshold,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help=f"Confirm that you have read and accept the MovieLens terms at {TERMS_URL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/movielens"),
        help="Ignored local directory for the archive, source file, and converted CSV.",
    )
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing local ml-latest-small.zip instead of downloading (useful offline).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.accept_terms:
        raise SystemExit(
            "Refusing to download or transform MovieLens without --accept-terms. "
            f"Read {TERMS_URL} first."
        )
    if not 0.5 <= args.positive_threshold <= 5.0:
        raise SystemExit("--positive-threshold must be between 0.5 and 5.0")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ml-latest-small.zip"
    if args.archive:
        shutil.copyfile(args.archive, archive_path)
    else:
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, archive_path)

    ratings_path = output_dir / "ratings.csv"
    with zipfile.ZipFile(archive_path) as archive:
        if ARCHIVE_MEMBER not in archive.namelist():
            raise SystemExit(f"Archive does not contain {ARCHIVE_MEMBER}")
        with archive.open(ARCHIVE_MEMBER) as source, ratings_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    interactions_path = output_dir / "interactions.csv"
    counts = convert_ratings(
        ratings_path,
        interactions_path,
        positive_threshold=args.positive_threshold,
    )
    metadata = {
        "dataset": "MovieLens latest-small",
        "purpose": "optional generic recommendation benchmark; not dating-app evidence",
        "source_url": SOURCE_URL,
        "terms_url": TERMS_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "output_sha256": sha256_file(interactions_path),
        "redistribution": "not redistributed by this repository; downloaded locally by the user",
        **counts,
    }
    metadata_path = output_dir / "provenance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Run: python recommender.py --csv {interactions_path} evaluate --top-k 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
