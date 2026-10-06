from pathlib import Path
import pytest

from recommender import (
    build_model,
    evaluate_ranker,
    load_likes,
    temporal_split,
    top_k_for_user,
    top_k_popularity,
    top_k_random,
    evaluate,
    build_parser,
)


SAMPLE = Path("examples/sample_swipes.csv")


def test_sample_data_loads_positive_interactions_only():
    likes = load_likes(SAMPLE)

    assert len(likes) == 9
    assert likes["like"].eq(1).all()
    assert likes["decidermemberid"].nunique() == 3


@pytest.mark.parametrize("cutoff", [0, -1])
def test_invalid_cutoffs_cannot_produce_rankings_or_metrics(cutoff):
    train, holdout = temporal_split(load_likes(SAMPLE))
    model = build_model(train, components=2)
    for rank in (top_k_for_user, top_k_popularity, top_k_random):
        with pytest.raises(ValueError, match="positive integer"):
            rank(model, "u1", cutoff)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_ranker(model, holdout, cutoff, ranker="svd")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--top-k", str(cutoff)])


def test_output_discloses_per_user_split_is_not_prospective(capsys):
    train, holdout = temporal_split(load_likes(SAMPLE))
    evaluate(build_model(train, components=2), holdout, 2)
    assert "not a globally prospective evaluation" in capsys.readouterr().out


def test_model_can_score_sample_user():
    likes = load_likes(SAMPLE)
    train, _ = temporal_split(likes)
    model = build_model(train, components=2)

    recommendations = top_k_for_user(model, "u1", top_k=2)

    assert recommendations
    assert all(isinstance(candidate, str) for candidate, _ in recommendations)
    assert all(isinstance(score, float) for _, score in recommendations)


def test_evaluation_compares_personalisation_with_popularity():
    likes = load_likes(SAMPLE)
    train, test = temporal_split(likes)
    model = build_model(train, components=2)

    personalised = evaluate_ranker(model, test, 2, ranker="svd")
    baseline = evaluate_ranker(model, test, 2, ranker="popularity")

    assert personalised.evaluated_users == baseline.evaluated_users == 3
    assert 0 <= personalised.hit_rate <= 1
    assert 0 <= baseline.hit_rate <= 1
    assert 0 <= personalised.recommendation_coverage <= 1
    assert top_k_popularity(model, "u1", 2)


def test_catalog_missing_holdout_is_counted_as_a_miss():
    likes = load_likes(SAMPLE)
    train, test = temporal_split(likes)
    model = build_model(train, components=2)
    missing = test.iloc[[0]].copy()
    missing.loc[:, "othermemberid"] = "new-profile"

    result = evaluate_ranker(model, missing, 2, ranker="svd")

    assert result.evaluated_users == 1
    assert result.catalog_known_holdouts == 0
    assert result.hits == 0
    assert result.total_holdouts == 1
    assert result.skipped_reasons == {}
    assert result.constraint_reasons == {"item_not_in_train": 1}


def test_cold_user_gets_declared_popularity_fallback_and_random_is_repeatable():
    likes = load_likes(SAMPLE)
    train, test = temporal_split(likes)
    model = build_model(train, components=2)
    cold = test.iloc[[0]].copy()
    cold.loc[:, "decidermemberid"] = "new-user"

    personalised = evaluate_ranker(model, cold, 2, ranker="svd")
    random_a = evaluate_ranker(model, test, 2, ranker="random")
    random_b = evaluate_ranker(model, test, 2, ranker="random")

    assert personalised.evaluated_users == 1
    assert personalised.skipped_reasons == {}
    assert personalised.constraint_reasons == {"user_not_in_train": 1}
    assert personalised.cold_start_fallbacks == 1
    assert random_a == random_b
