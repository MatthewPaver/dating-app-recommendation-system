# Recommender Evaluation Lab: does personalisation beat the baselines?

An offline evaluation lab for implicit-feedback recommenders: it ranks items for each user with truncated SVD and checks, on a per-user temporal holdout, whether that beats popularity and random baselines, counting cold-start cases instead of hiding them.

[![Validate](https://github.com/MatthewPaver/recommender-eval-lab/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/recommender-eval-lab/actions/workflows/validate.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3670A0.svg)

## Result: SVD beats both baselines on the bundled synthetic set

Bundled deterministic generator (100 users, 160 items, 30 interactions per user, seed 42), each user's latest like held out, K=10:

| Ranker | Hits / 100 | HitRate@10 | MRR@10 | Catalogue coverage |
| --- | --- | --- | --- | --- |
| Random (deterministic) | 6 | 0.0600 | 0.0177 | 1.0000 |
| Popularity | 9 | 0.0900 | 0.0255 | 0.0938 |
| Truncated SVD (32 factors) | 14 | 0.1400 | 0.0409 | 0.9625 |

Personalisation lift vs popularity: **+0.05 HitRate@10**. All 100 holdout users and items are in the train catalogue, so no cold-start fallbacks fire on this set.

Produced by:

```bash
.venv/bin/python synthetic_swipes.py --output data/synthetic_swipes.csv
.venv/bin/python recommender.py --csv data/synthetic_swipes.csv evaluate --top-k 10
```

What this does **not** show: the synthetic users have a planted preference (each likes two of eight item clusters with probability 0.72, others 0.20), so SVD winning here shows the pipeline can detect a real signal, not that SVD will win on your data. These are point estimates from one split and one seed, with no confidence interval; 14 against 9 hits out of 100 has not been tested for significance. The 10-row sample in `examples/` is a smoke test and gives a three-way tie (below).

## The problem

Personalised ranking is expensive to build and run, and the usual offline check is flattering: score the model alone, drop users and items it has never seen, and report a number with nothing to compare it against. A popularity list often gets most of the way for a fraction of the cost.

This lab asks one question before you build personalisation into a product: on your own interaction history, does it beat the simple baselines on the same holdouts? A zero lift is a valid answer. The bundled example uses a swipe-style dataset (user, item, timestamp, like), but the same shape fits marketplace shortlists, content or job recommendations, or any "what should we show next?" list driven by positive interactions.

<a id="quick-start"></a>

## Quickstart

Python 3.11+ (CI uses 3.11):

```bash
git clone https://github.com/MatthewPaver/recommender-eval-lab.git
cd recommender-eval-lab
make demo PYTHON=python3.11          # installs CLI deps, evaluates and queries the 10-row sample
make synthetic && .venv/bin/python recommender.py --csv data/synthetic_swipes.csv evaluate --top-k 10
```

The last command prints the results table above. `make demo` on the tiny sample prints:

```text
Temporal holdout evaluation @ 2
Per-user latest-event split: not a globally prospective evaluation; other users' training events may be later.
Total holdouts: 3
Skipped holdouts: 0; skipped reasons: none (cold-start cases are scored as misses/fallbacks)
ranker      users  known_users  known_items  hits  hit_rate  mrr     rec_coverage
random          3            3            1     1    0.3333  0.3333        0.6667
  constraints: {'item_not_in_train': 2}; cold-start fallbacks: 0
popularity      3            3            1     1    0.3333  0.3333        0.6667
  constraints: {'item_not_in_train': 2}; cold-start fallbacks: 0
svd             3            3            1     1    0.3333  0.3333        0.6667
  constraints: {'item_not_in_train': 2}; cold-start fallbacks: 0
Personalisation lift vs popularity (HitRate@2): +0.0000
Latent dimensions used: 2
Top 1 recommendations for user u1
 1. p4  score=0.0000
```

Two of the three held-out items never appear in training, so every ranker must miss them; they are counted as misses, not dropped. Exit codes: `0` success, `1` input or data error (with the CSV row and field to fix), `2` invalid arguments.

### Bring your own interactions

Use pseudonymous IDs and these four columns; other columns are ignored:

```csv
decidermemberid,othermemberid,timestamp,like
001,article-7,2026-01-01T09:00:00Z,1
001,article-8,2026-01-02T09:00:00Z,0
```

`decidermemberid` is the user and `othermemberid` the item. `like` must be 0 or 1; timestamps must be ISO dates or datetimes (naive values are read as UTC). IDs stay as text, including leading zeroes. Blank IDs, invalid dates and other `like` values stop the command with the row and field to fix, rather than silently changing your data.

```bash
.venv/bin/python recommender.py --csv /path/to/interactions.csv summary
.venv/bin/python recommender.py --csv /path/to/interactions.csv evaluate --top-k 5
.venv/bin/python recommender.py --csv /path/to/interactions.csv recommend --user-id 001 --top-k 5
```

Run `summary` first: it accounts for input rows, ignored negatives, duplicate positives (the latest timestamp is kept) and retained positives. SVD needs at least two users and two liked items, and evaluation needs history left after the holdout. Everything runs locally; the CSV is never uploaded or modified. Keep private data outside the repository.

On Windows without Make: `py -3.11 -m venv .venv`, `.venv\Scripts\python.exe -m pip install -r requirements-cli.txt`, then the commands above with `.venv\Scripts\python.exe`.

### Optional: MovieLens and the notebook

```bash
make movielens   # downloads ml-latest-small from GroupLens; read their terms first
.venv/bin/python recommender.py --csv data/movielens/interactions.csv evaluate --top-k 10
make notebook    # installs Jupyter, generates the synthetic set, opens the walkthrough
```

The MovieLens downloader requires `--accept-terms` (set by `make movielens`), maps ratings of 4.0 or above to positive interactions and records SHA-256 provenance in `data/movielens/provenance.json`. The data is git-ignored and not redistributed. Read the [GroupLens terms](https://grouplens.org/datasets/movielens/) before running it.

## How it works

```mermaid
flowchart LR
    CSV[(Interactions CSV<br/>user, item, timestamp, like)] --> V[Validate rows<br/>keep likes, dedupe]
    V --> S{Per-user split}
    S -- latest like --> H[Holdout]
    S -- earlier likes --> M[Sparse user x item matrix]
    M --> SVD[Truncated SVD<br/>up to 32 factors]
    M --> POP[Popularity baseline]
    M --> RND[Deterministic random baseline]
    SVD --> K[Top-K per user<br/>seen items excluded]
    POP --> K
    RND --> K
    K --> E[HitRate@K, MRR@K,<br/>coverage, cold-start counts]
    H --> E
```

- **Loading and validation** (`recommender.py: load_likes`): checks every row, keeps `like = 1`, deduplicates repeated user/item pairs and reports what it dropped.
- **Split** (`temporal_split`): holds out each user's most recent like; everything earlier is training.
- **Model** (`build_model`, `top_k_for_user`): a binary sparse matrix factorised with scikit-learn's `TruncatedSVD` (`random_state=42`); items the user already liked are excluded from their ranking.
- **Baselines** (`top_k_popularity`, `top_k_random`): train-set popularity, and a random order fixed by hashing (user, item), both over the same catalogue with the same seen-item filter.
- **Evaluation** (`evaluate_ranker`): HitRate@K, MRR@K, catalogue coverage, and counts of holdouts whose user or item is not in training.
- **Data** (`synthetic_swipes.py`, `scripts/download_movielens.py`): the deterministic fictional generator and the opt-in public benchmark adapter.

## Results

The table at the top is the headline. How to read the columns:

- **Hits / HitRate@K:** the share of users whose held-out item appears in their top K.
- **MRR@K:** the mean of 1 / rank of the held-out item (0 when it is outside the top K), so it rewards ranking it near the top.
- **Catalogue coverage:** the share of training items that appear in anyone's top K. On the synthetic set, popularity's 0.0938 is 15 of 160 items: almost everyone sees the same list. SVD's 0.9625 is 154 of 160, so its lists differ by user.
- **known_users / known_items:** how many holdouts the model could have got right at all. When a held-out item never appears in training, every ranker scores a miss, and the output lists it under `constraints`.

No MovieLens result is reported yet; the adapter is there so the same comparison can be run on a recognised public dataset.

## Design decisions and trade-offs

- **Baselines share the whole pipeline.** Random, popularity and SVD use the same training catalogue, the same seen-item filter and the same holdouts, so the lift is like for like. The cost: both baselines are weak. A stronger one, such as item-to-item nearest neighbours, is not included, so beating them is a necessary bar, not a sufficient one.
- **Truncated SVD rather than weighted ALS or BPR.** It needs only scikit-learn, installs in seconds in CI and is deterministic with a fixed seed. The cost: it treats every unobserved cell the same and is not optimised for implicit feedback the way weighted ALS or BPR are, so it understates what a production recommender could do.
- **Per-user latest-like holdout rather than one global time cutoff.** Every user contributes one holdout, which matters on small data. The cost: another user's training interactions can be later than a holdout, so this is not a prospective simulation. The CLI prints that caveat with every result.
- **Cold start is counted, not filtered.** Held-out items that never appear in training are scored as misses, and unknown users get a labelled popularity fallback. Dropping them would raise every number; counting them keeps the figures honest about what the model could ever recommend.
- **Passes are discarded, not treated as negatives.** Only likes enter the matrix, so a pass looks the same as an item the user never saw. This avoids learning from noisy negatives, but throws away whatever signal a pass carries.
- **Bad input stops the run.** Invalid rows exit `1` with the row and field to fix rather than being coerced or skipped, so a result always describes the data you supplied.

## Limits and non-goals

- The synthetic set has a planted preference structure, so the SVD win on it is expected by construction. It validates the pipeline, not the method on real data.
- Results are point estimates from one split and one seed. There are no confidence intervals or significance tests.
- The bundled sample (10 rows, 3 users) is a smoke test. Its tie is not evidence either way.
- Offline only: no online feedback loop, A/B test, serving path, or fairness analysis.
- The swipe-shaped example is a familiar implicit-feedback format, not a dating product. Scores are ranking values, not probabilities of compatibility or satisfaction, and nothing here supports claims about matching outcomes.
- The notebook (`recommendation_system_walkthrough.ipynb`) re-implements the pipeline for narrative and is committed without outputs. `recommender.py` is the source of truth for every number above.
- Dependencies are lower-bounded (`>=`), not locked.

## Repository layout and tests

```text
recommender.py                            CLI: summary, evaluate, recommend (SVD and baselines)
synthetic_swipes.py                       Deterministic fictional-data generator
scripts/download_movielens.py             Opt-in MovieLens adapter with provenance
examples/sample_swipes.csv                10-row fictional smoke-test dataset
recommendation_system_walkthrough.ipynb   Narrative notebook on the synthetic set
tests/                                    pytest: input validation, rankers, generator, adapter
Makefile                                  demo, test, synthetic, movielens, notebook
requirements-cli.txt / requirements.txt   CLI and test deps / adds the notebook stack
```

```bash
make test        # pytest tests -q
```

CI ([`.github/workflows/validate.yml`](.github/workflows/validate.yml)) runs the tests and `make demo` on Python 3.11 for every push and pull request. All committed data is fictional; no personal dataset is distributed.

## Licence

Code and synthetic examples: MIT. See [LICENSE](LICENSE).
