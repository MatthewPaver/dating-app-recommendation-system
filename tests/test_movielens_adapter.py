import csv
import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/download_movielens.py")
SPEC = importlib.util.spec_from_file_location("download_movielens", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_movielens_adapter_keeps_only_thresholded_ratings(tmp_path):
    source = tmp_path / "ratings.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["userId", "movieId", "rating", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"userId": "1", "movieId": "10", "rating": "5.0", "timestamp": "1"},
                {"userId": "1", "movieId": "11", "rating": "3.5", "timestamp": "2"},
                {"userId": "2", "movieId": "10", "rating": "4.0", "timestamp": "3"},
            ]
        )

    output = tmp_path / "interactions.csv"
    result = MODULE.convert_ratings(source, output, positive_threshold=4.0)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert result == {
        "ratings_read": 3,
        "positive_interactions": 2,
        "users": 2,
        "items": 1,
        "positive_threshold": 4.0,
    }
    assert [row["decidermemberid"] for row in rows] == ["ml-user-1", "ml-user-2"]
    assert all(row["like"] == "1" for row in rows)
    assert rows[0]["timestamp"] == "1970-01-01T00:00:01+00:00"
