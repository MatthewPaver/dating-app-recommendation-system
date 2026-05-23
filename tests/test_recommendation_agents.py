from recommendation_agents import (
    diversity_agent,
    explanation_agent,
    fairness_agent,
    run_recommendation_agent_review,
)


RECS = [
    {"id": "p1", "score": 0.9, "group": "a", "cluster": "outdoors", "reasons": ["Similar swipe history"]},
    {"id": "p2", "score": 0.8, "group": "b", "cluster": "music", "reasons": ["Strong latent match"]},
    {"id": "p3", "score": 0.7, "group": "b", "cluster": "food", "reasons": ["Popular with similar users"]},
]


def test_fairness_agent_flags_underrepresented_groups():
    signal = fairness_agent([{"group": "a"}, {"group": "a"}, {"group": "b"}], min_share=0.4)

    assert signal.status == "review"
    assert any("b:" in item for item in signal.evidence)


def test_diversity_agent_passes_varied_shortlist():
    assert diversity_agent(RECS).status == "pass"


def test_explanation_agent_requires_reasons():
    assert explanation_agent({"score": 0.4}).status == "review"
    assert explanation_agent(RECS[0]).status == "pass"


def test_review_uses_three_agents():
    assert [note.agent for note in run_recommendation_agent_review(RECS)] == [
        "fairness_agent",
        "diversity_agent",
        "explanation_agent",
    ]
