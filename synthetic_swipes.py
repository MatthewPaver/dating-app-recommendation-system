#!/usr/bin/env python3
"""Generate deterministic synthetic swipe interactions for demos and notebooks."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


FIELDNAMES = [
    "decidermemberid",
    "othermemberid",
    "timestamp",
    "like",
    "decidergender",
    "othergender",
    "deciderdobyear",
    "otherdobyear",
    "decidersignuptimestamp",
    "othersignuptimestamp",
]


def generate_rows(
    users: int = 100,
    profiles: int = 160,
    interactions_per_user: int = 30,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return reproducible, entirely fictional swipe rows."""
    if users < 2:
        raise ValueError("users must be at least 2")
    if profiles < 3:
        raise ValueError("profiles must be at least 3")
    if not 2 <= interactions_per_user <= profiles:
        raise ValueError("interactions_per_user must be between 2 and profiles")

    rng = random.Random(seed)
    start = datetime(2026, 1, 1, 9, 0, 0)
    genders = ("F", "M", "X")
    rows: list[dict[str, object]] = []

    profile_metadata = {
        f"p{index:04d}": {
            "gender": genders[index % len(genders)],
            "dobyear": 1984 + (index % 20),
            "signup": start - timedelta(days=30 + (index % 60)),
            "cluster": index % 8,
        }
        for index in range(1, profiles + 1)
    }
    profile_ids = list(profile_metadata)

    for user_index in range(1, users + 1):
        user_id = f"u{user_index:04d}"
        user_gender = genders[(user_index + 1) % len(genders)]
        user_birth_year = 1984 + ((user_index * 3) % 20)
        user_signup = start - timedelta(days=45 + (user_index % 45))
        preferred_cluster = user_index % 8
        seen_profiles = rng.sample(profile_ids, interactions_per_user)

        for interaction_index, profile_id in enumerate(seen_profiles):
            profile = profile_metadata[profile_id]
            preference_match = profile["cluster"] in {
                preferred_cluster,
                (preferred_cluster + 1) % 8,
            }
            like_probability = 0.72 if preference_match else 0.20
            liked = int(rng.random() < like_probability)

            # Guarantee enough positive history for every synthetic user.
            if interaction_index < 2:
                liked = 1

            timestamp = start + timedelta(
                minutes=(user_index * interactions_per_user) + interaction_index
            )
            rows.append(
                {
                    "decidermemberid": user_id,
                    "othermemberid": profile_id,
                    "timestamp": timestamp.isoformat(sep=" "),
                    "like": liked,
                    "decidergender": user_gender,
                    "othergender": profile["gender"],
                    "deciderdobyear": user_birth_year,
                    "otherdobyear": profile["dobyear"],
                    "decidersignuptimestamp": user_signup.isoformat(sep=" "),
                    "othersignuptimestamp": profile["signup"].isoformat(sep=" "),
                }
            )

    return rows


def write_csv(output: Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_swipes.csv"))
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--profiles", type=int, default=160)
    parser.add_argument("--interactions-per-user", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = generate_rows(
        users=args.users,
        profiles=args.profiles,
        interactions_per_user=args.interactions_per_user,
        seed=args.seed,
    )
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} synthetic interactions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
