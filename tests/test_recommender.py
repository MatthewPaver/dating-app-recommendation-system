from pathlib import Path

from recommender import build_model, load_likes, temporal_split, top_k_for_user


SAMPLE = Path("examples/sample_swipes.csv")


def test_sample_data_loads_positive_interactions_only():
    likes = load_likes(SAMPLE)

    assert len(likes) == 9
    assert likes["like"].eq(1).all()
    assert likes["decidermemberid"].nunique() == 3


def test_model_can_score_sample_user():
    likes = load_likes(SAMPLE)
    train, _ = temporal_split(likes)
    model = build_model(train, components=2)

    recommendations = top_k_for_user(model, "u1", top_k=2)

    assert recommendations
    assert all(isinstance(candidate, str) for candidate, _ in recommendations)
    assert all(isinstance(score, float) for _, score in recommendations)
