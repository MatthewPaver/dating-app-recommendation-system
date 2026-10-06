# Audit fixes — 2026-09-05

This repository remains an **offline recommender lab**, not evidence of improved compatibility or dating outcomes.

## Changes

- Evaluation denominators now include every holdout presented to a ranker. Train-unknown users and items are reported separately instead of silently disappearing.
- Train-unknown users have a declared cold-start policy: SVD falls back to popularity. Train-unknown held-out items remain misses because no train-catalog ranker can retrieve them.
- SVD is compared with popularity and deterministic-random baselines under the same holdouts, train catalogue and unseen-item filter.
- Output includes total holdouts, evaluated users, known-user/item counts, constraint reasons, cold-start fallbacks and recommendation coverage.
- CLI/test dependencies are separated from optional notebook dependencies.
- An opt-in MovieLens latest-small installer uses the official GroupLens download, requires explicit terms acknowledgement and writes checksum provenance. MovieLens is a generic permitted benchmark, not dating evidence.

## Verification

- Fresh Python 3.11 environment: wheel-only `requirements-cli.txt` installation succeeded; full suite **12 passed**. The sample evaluation also completed in that environment.
- Independent review reproduced invalid zero/negative cutoffs producing rankings; all rankers, evaluation and both CLI commands now reject them. Three new regression cases failed before the fix and passed afterwards, including the missing per-user split caveat in CLI output.
- Sample evaluation completed with three holdouts: only one held-out item was present in the train catalogue; SVD, popularity and random each scored HitRate@2 `0.3333`; SVD lift over popularity was `+0.0000`.

## Remaining limitations

- The included ten-interaction fixture is only a smoke test. Ranking-quality conclusions require the opt-in larger dataset and repeated/global temporal evaluation appropriate to the deployment decision.
- The current split is each user's latest interaction, not a single global cutoff; the CLI labels this limitation.
- Clean installation used Python 3.11 with pandas 3.0.5, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0. This verifies the current dependency resolution on macOS, not every platform or future resolution of the version ranges.
- The optional MovieLens network download was not executed; its conversion logic was tested against a local fixture and requires the user to accept GroupLens terms.
