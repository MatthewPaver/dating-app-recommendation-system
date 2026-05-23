from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AgentNote:
    agent: str
    status: str
    message: str
    evidence: list[str]


def fairness_agent(recommendations: Iterable[dict], protected_key: str = "group", min_share: float = 0.2) -> AgentNote:
    rows = list(recommendations)
    if not rows:
        return AgentNote("fairness_agent", "review", "No recommendations to inspect", [],)

    counts: dict[str, int] = {}
    for row in rows:
        group = str(row.get(protected_key, "unknown"))
        counts[group] = counts.get(group, 0) + 1

    shares = {group: count / len(rows) for group, count in counts.items()}
    underrepresented = [group for group, share in shares.items() if share < min_share]
    return AgentNote(
        "fairness_agent",
        "review" if underrepresented else "pass",
        "Recommendation mix needs fairness review" if underrepresented else "Recommendation mix passes simple share check",
        [f"{group}: {share:.0%}" for group, share in sorted(shares.items())],
    )


def diversity_agent(recommendations: Iterable[dict], attribute: str = "cluster", min_unique: int = 3) -> AgentNote:
    rows = list(recommendations)
    unique_values = {row.get(attribute, "unknown") for row in rows}
    return AgentNote(
        "diversity_agent",
        "pass" if len(unique_values) >= min_unique else "review",
        f"{len(unique_values)} unique {attribute} values in shortlist",
        [str(value) for value in sorted(unique_values)],
    )


def explanation_agent(recommendation: dict) -> AgentNote:
    reasons = recommendation.get("reasons") or []
    score = float(recommendation.get("score", 0.0))
    evidence = [f"score={score:.4f}", *[str(reason) for reason in reasons[:3]]]
    return AgentNote(
        "explanation_agent",
        "pass" if reasons else "review",
        "Recommendation has a user-safe explanation" if reasons else "Recommendation lacks explanation evidence",
        evidence,
    )


def run_recommendation_agent_review(recommendations: Iterable[dict]) -> list[AgentNote]:
    rows = list(recommendations)
    first = rows[0] if rows else {}
    return [
        fairness_agent(rows),
        diversity_agent(rows),
        explanation_agent(first),
    ]
