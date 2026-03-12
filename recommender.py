#!/usr/bin/env python3
"""Lightweight CLI for training and querying the dating app recommender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


REQUIRED_COLUMNS = {"decidermemberid", "othermemberid", "timestamp", "like"}


def load_likes(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. If you just cloned the repo, run 'git lfs pull'."
        )

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    likes = frame.loc[:, ["decidermemberid", "othermemberid", "timestamp", "like"]].copy()
    likes["decidermemberid"] = likes["decidermemberid"].astype(str)
    likes["othermemberid"] = likes["othermemberid"].astype(str)
    likes["like"] = pd.to_numeric(likes["like"], errors="coerce").fillna(0).astype(int)
    likes["timestamp"] = pd.to_datetime(likes["timestamp"], utc=True, errors="coerce")
    likes = likes[(likes["like"] == 1) & likes["timestamp"].notna()].copy()
    likes.sort_values(["decidermemberid", "timestamp"], inplace=True)
    # Treat repeated likes on the same profile as one positive interaction.
    likes = likes.drop_duplicates(["decidermemberid", "othermemberid"], keep="last")

    if likes.empty:
        raise ValueError("No positive interactions were found in the provided CSV.")

    return likes


def temporal_split(likes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_idx = likes.groupby("decidermemberid")["timestamp"].idxmax()
    test = likes.loc[latest_idx].copy()
    train = likes.drop(latest_idx).copy()
    test = test[test["decidermemberid"].isin(train["decidermemberid"])]

    if train.empty:
        raise ValueError("Not enough history to build a train split after temporal holdout.")

    return train, test


def build_model(train: pd.DataFrame, components: int) -> dict[str, object]:
    users = pd.Index(sorted(train["decidermemberid"].unique()))
    items = pd.Index(sorted(train["othermemberid"].unique()))

    user_codes = users.get_indexer(train["decidermemberid"])
    item_codes = items.get_indexer(train["othermemberid"])
    matrix = csr_matrix(
        (np.ones(len(train), dtype=np.float32), (user_codes, item_codes)),
        shape=(len(users), len(items)),
    )

    min_dim = min(matrix.shape)
    if min_dim <= 1:
        raise ValueError("Interaction matrix is too small to factorise.")

    n_components = max(1, min(components, min_dim - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T

    return {
        "matrix": matrix,
        "users": users,
        "items": items,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "components": n_components,
    }


def top_k_for_user(model: dict[str, object], user_id: str, top_k: int) -> list[tuple[str, float]]:
    users: pd.Index = model["users"]  # type: ignore[assignment]
    items: pd.Index = model["items"]  # type: ignore[assignment]
    matrix: csr_matrix = model["matrix"]  # type: ignore[assignment]
    user_factors: np.ndarray = model["user_factors"]  # type: ignore[assignment]
    item_factors: np.ndarray = model["item_factors"]  # type: ignore[assignment]

    if user_id not in users:
        raise KeyError(f"Unknown user id: {user_id}")

    user_idx = users.get_loc(user_id)
    scores = user_factors[user_idx] @ item_factors.T
    scores = np.asarray(scores, dtype=float)
    seen_items = matrix[user_idx].indices
    scores[seen_items] = -np.inf

    valid = np.isfinite(scores)
    if not valid.any():
        return []

    candidate_count = min(top_k, int(valid.sum()))
    top_idx = np.argpartition(scores, -candidate_count)[-candidate_count:]
    ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(items[idx], float(scores[idx])) for idx in ordered]


def print_summary(likes: pd.DataFrame) -> None:
    print("Dataset summary")
    print(f"Users: {likes['decidermemberid'].nunique()}")
    print(f"Profiles: {likes['othermemberid'].nunique()}")
    print(f"Positive interactions: {len(likes)}")
    print(f"Date range: {likes['timestamp'].min()} -> {likes['timestamp'].max()}")


def evaluate(model: dict[str, object], test: pd.DataFrame, top_k: int) -> None:
    users: pd.Index = model["users"]  # type: ignore[assignment]
    items: pd.Index = model["items"]  # type: ignore[assignment]

    hits = 0
    reciprocal_ranks: list[float] = []
    evaluated = 0

    for row in test.itertuples(index=False):
        user_id = str(row.decidermemberid)
        item_id = str(row.othermemberid)
        if user_id not in users or item_id not in items:
            continue

        ranked = top_k_for_user(model, user_id, top_k)
        if not ranked:
            continue

        evaluated += 1
        ranked_ids = [candidate for candidate, _ in ranked]
        if item_id in ranked_ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (ranked_ids.index(item_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

    if evaluated == 0:
        raise ValueError("No evaluable holdout rows remained after filtering to train-known users/items.")

    print("Evaluation")
    print(f"Evaluated users: {evaluated}")
    print(f"HitRate@{top_k}: {hits / evaluated:.4f}")
    print(f"MRR@{top_k}: {np.mean(reciprocal_ranks):.4f}")
    print(f"Latent dimensions used: {model['components']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("swipes.csv"), help="Path to the swipe dataset CSV.")
    parser.add_argument("--components", type=int, default=32, help="Maximum latent dimensions for TruncatedSVD.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print a quick dataset summary.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Train the model and print temporal holdout metrics.")
    evaluate_parser.add_argument("--top-k", type=int, default=10, help="Recommendation cutoff for metrics.")

    recommend_parser = subparsers.add_parser("recommend", help="Train the model and print recommendations for one user.")
    recommend_parser.add_argument("--user-id", required=True, help="User identifier to score.")
    recommend_parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to print.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        likes = load_likes(args.csv)
        if args.command == "summary":
            print_summary(likes)
            return 0

        train, test = temporal_split(likes)
        model = build_model(train, args.components)

        if args.command == "evaluate":
            evaluate(model, test, args.top_k)
            return 0

        if args.command == "recommend":
            results = top_k_for_user(model, args.user_id, args.top_k)
            if not results:
                print("No unseen candidates available for this user.")
                return 0

            print(f"Top {len(results)} recommendations for user {args.user_id}")
            for rank, (candidate, score) in enumerate(results, start=1):
                print(f"{rank:>2}. {candidate}  score={score:.4f}")
            return 0

        parser.error("Unhandled command")
        return 2
    except Exception as exc:  # pragma: no cover - lightweight CLI surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
