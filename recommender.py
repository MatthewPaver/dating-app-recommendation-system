#!/usr/bin/env python3
"""CLI for evaluating an implicit-feedback recommender against popularity and random baselines."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


REQUIRED_COLUMNS = {"decidermemberid", "othermemberid", "timestamp", "like"}


@dataclass(frozen=True)
class RankingMetrics:
    """Offline ranking results with the coverage caveats kept visible."""

    ranker: str
    total_holdouts: int
    evaluated_users: int
    train_known_users: int
    catalog_known_holdouts: int
    skipped_reasons: dict[str, int]
    constraint_reasons: dict[str, int]
    cold_start_fallbacks: int
    hits: int
    hit_rate: float
    mrr: float
    recommendation_coverage: float


def load_likes(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Run 'make synthetic' to generate demo data."
        )

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    likes = frame.loc[:, ["decidermemberid", "othermemberid", "timestamp", "like"]].copy()
    def reject_rows(mask: pd.Series, field: str, requirement: str) -> None:
        if mask.any():
            rows = ", ".join(str(index + 2) for index in likes.index[mask][:5])
            raise ValueError(f"CSV row(s) {rows}: {field} {requirement}. Fix the input and retry.")

    for field in ("decidermemberid", "othermemberid"):
        likes[field] = likes[field].str.strip()
        reject_rows(likes[field].eq(""), field, "must contain a non-blank identifier")
    likes["like"] = pd.to_numeric(likes["like"], errors="coerce")
    reject_rows(~likes["like"].isin([0, 1]), "like", "must be 0 or 1")
    likes["timestamp"] = pd.to_datetime(likes["timestamp"], utc=True, errors="coerce", format="ISO8601")
    reject_rows(likes["timestamp"].isna(), "timestamp", "must be an ISO date or datetime")
    negative_rows = int(likes["like"].eq(0).sum())
    likes = likes[likes["like"] == 1].copy()
    positive_rows = len(likes)
    likes.sort_values(["decidermemberid", "timestamp"], inplace=True)
    # Treat repeated likes on the same profile as one positive interaction.
    likes = likes.drop_duplicates(["decidermemberid", "othermemberid"], keep="last")

    if likes.empty:
        raise ValueError("No positive interactions were found in the provided CSV.")

    likes.attrs["input_summary"] = {
        "input_rows": len(frame), "negative_rows": negative_rows,
        "duplicate_positive_rows": positive_rows - len(likes), "retained_positive_rows": len(likes),
    }
    return likes


def temporal_split(likes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_idx = likes.groupby("decidermemberid")["timestamp"].idxmax()
    test = likes.loc[latest_idx].copy()
    train = likes.drop(latest_idx).copy()
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
        "popularity": train["othermemberid"].value_counts().to_dict(),
    }


def validate_cutoff(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top-k must be a positive integer")
    return top_k


def parse_cutoff(value: str) -> int:
    try:
        return validate_cutoff(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be a positive integer") from exc


def top_k_for_user(model: dict[str, object], user_id: str, top_k: int) -> list[tuple[str, float]]:
    validate_cutoff(top_k)
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


def top_k_popularity(model: dict[str, object], user_id: str, top_k: int) -> list[tuple[str, float]]:
    """Rank globally popular unseen items as a deliberately simple baseline."""
    validate_cutoff(top_k)

    users: pd.Index = model["users"]  # type: ignore[assignment]
    items: pd.Index = model["items"]  # type: ignore[assignment]
    matrix: csr_matrix = model["matrix"]  # type: ignore[assignment]
    popularity: dict[str, int] = model["popularity"]  # type: ignore[assignment]
    seen: set[str] = set()
    if user_id in users:
        user_idx = users.get_loc(user_id)
        seen = {str(items[index]) for index in matrix[user_idx].indices}
    candidates = [
        (str(item_id), float(popularity.get(str(item_id), 0)))
        for item_id in items
        if str(item_id) not in seen
    ]
    candidates.sort(key=lambda row: (-row[1], row[0]))
    return candidates[:top_k]


def top_k_random(model: dict[str, object], user_id: str, top_k: int) -> list[tuple[str, float]]:
    """Return a deterministic random unseen ranking for a fair smoke-test baseline."""
    validate_cutoff(top_k)
    users: pd.Index = model["users"]  # type: ignore[assignment]
    items: pd.Index = model["items"]  # type: ignore[assignment]
    matrix: csr_matrix = model["matrix"]  # type: ignore[assignment]
    seen: set[str] = set()
    if user_id in users:
        user_idx = users.get_loc(user_id)
        seen = {str(items[index]) for index in matrix[user_idx].indices}
    candidates = [str(item) for item in items if str(item) not in seen]
    candidates.sort(key=lambda item: hashlib.sha256(f"{user_id}\0{item}".encode()).digest())
    return [(item, 0.0) for item in candidates[:top_k]]


def print_summary(likes: pd.DataFrame) -> None:
    print("Dataset summary")
    for key, value in likes.attrs.get("input_summary", {}).items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Users: {likes['decidermemberid'].nunique()}")
    print(f"Profiles: {likes['othermemberid'].nunique()}")
    print(f"Positive interactions: {len(likes)}")
    print(f"Date range: {likes['timestamp'].min()} -> {likes['timestamp'].max()}")


def evaluate_ranker(
    model: dict[str, object],
    test: pd.DataFrame,
    top_k: int,
    *,
    ranker: str,
) -> RankingMetrics:
    validate_cutoff(top_k)
    users: pd.Index = model["users"]  # type: ignore[assignment]
    items: pd.Index = model["items"]  # type: ignore[assignment]

    hits = 0
    reciprocal_ranks: list[float] = []
    evaluated = 0
    train_known_users = 0
    catalog_known_holdouts = 0
    constraint_reasons: dict[str, int] = {}
    cold_start_fallbacks = 0
    recommended_items: set[str] = set()

    for row in test.itertuples(index=False):
        user_id = str(row.decidermemberid)
        item_id = str(row.othermemberid)
        evaluated += 1
        user_known = user_id in users
        if user_known:
            train_known_users += 1
        else:
            constraint_reasons["user_not_in_train"] = constraint_reasons.get("user_not_in_train", 0) + 1
        if item_id in items:
            catalog_known_holdouts += 1
        else:
            constraint_reasons["item_not_in_train"] = constraint_reasons.get("item_not_in_train", 0) + 1
        if ranker == "svd" and user_known:
            ranked = top_k_for_user(model, user_id, top_k)
        elif ranker == "random":
            ranked = top_k_random(model, user_id, top_k)
        else:
            ranked = top_k_popularity(model, user_id, top_k)
            cold_start_fallbacks += int(ranker == "svd" and not user_known)
        ranked_ids = [candidate for candidate, _ in ranked]
        recommended_items.update(ranked_ids)
        if item_id in ranked_ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (ranked_ids.index(item_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

    if evaluated == 0:
        raise ValueError("No evaluable holdout rows remained after filtering to train-known users/items.")

    return RankingMetrics(
        ranker=ranker,
        total_holdouts=int(len(test)),
        evaluated_users=evaluated,
        train_known_users=train_known_users,
        catalog_known_holdouts=catalog_known_holdouts,
        skipped_reasons={},
        constraint_reasons=constraint_reasons,
        cold_start_fallbacks=cold_start_fallbacks,
        hits=hits,
        hit_rate=hits / evaluated,
        mrr=float(np.mean(reciprocal_ranks)),
        recommendation_coverage=len(recommended_items) / len(items),
    )


def evaluate(model: dict[str, object], test: pd.DataFrame, top_k: int) -> None:
    results = [
        evaluate_ranker(model, test, top_k, ranker="random"),
        evaluate_ranker(model, test, top_k, ranker="popularity"),
        evaluate_ranker(model, test, top_k, ranker="svd"),
    ]
    print(f"Temporal holdout evaluation @ {top_k}")
    print("Per-user latest-event split: not a globally prospective evaluation; other users' training events may be later.")
    print(f"Total holdouts: {len(test)}")
    print("Skipped holdouts: 0; skipped reasons: none (cold-start cases are scored as misses/fallbacks)")
    print("ranker      users  known_users  known_items  hits  hit_rate  mrr     rec_coverage")
    for result in results:
        print(
            f"{result.ranker:<11} {result.evaluated_users:>5}  "
            f"{result.train_known_users:>11}  {result.catalog_known_holdouts:>11}  {result.hits:>4}  "
            f"{result.hit_rate:>8.4f}  {result.mrr:>6.4f}  "
            f"{result.recommendation_coverage:>12.4f}"
        )
        if result.constraint_reasons:
            print(f"  constraints: {result.constraint_reasons}; cold-start fallbacks: {result.cold_start_fallbacks}")
    lift = results[2].hit_rate - results[1].hit_rate
    print(f"Personalisation lift vs popularity (HitRate@{top_k}): {lift:+.4f}")
    print(f"Latent dimensions used: {model['components']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "examples/sample_swipes.csv",
        help="Path to a swipe dataset CSV.",
    )
    parser.add_argument("--components", type=int, default=32, help="Maximum latent dimensions for TruncatedSVD.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print a quick dataset summary.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Train the model and print temporal holdout metrics.")
    evaluate_parser.add_argument("--top-k", type=parse_cutoff, default=10, help="Positive recommendation cutoff for metrics.")

    recommend_parser = subparsers.add_parser("recommend", help="Train the model and print recommendations for one user.")
    recommend_parser.add_argument("--user-id", required=True, help="User identifier to score.")
    recommend_parser.add_argument("--top-k", type=parse_cutoff, default=10, help="Positive number of recommendations to print.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.components < 1:
        parser.error("--components must be a positive integer")

    try:
        likes = load_likes(args.csv)
        if args.command == "summary":
            print_summary(likes)
            return 0

        if args.command == "evaluate":
            train, test = temporal_split(likes)
            model = build_model(train, args.components)
            evaluate(model, test, args.top_k)
            return 0

        if args.command == "recommend":
            model = build_model(likes, args.components)
            if args.user_id in model["users"]:
                results = top_k_for_user(model, args.user_id, args.top_k)
            else:
                print("Unknown user: using the popularity baseline, not personalisation.")
                results = top_k_popularity(model, args.user_id, args.top_k)
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
