# Dating App Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
[![Validate](https://github.com/MatthewPaver/dating-app-recommendation-system/actions/workflows/validate.yml/badge.svg)](https://github.com/MatthewPaver/dating-app-recommendation-system/actions/workflows/validate.yml)

**User-user collaborative filtering for swipe-based dating apps**

*Notebook project with a lightweight CLI for evaluation and recommendation lookup*

</div>

---

## Portfolio Quick Read

| Section | Where to look |
|:---|:---|
| What it solves | Ranks unseen profiles from swipe-style implicit feedback instead of treating the dataset as static EDA |
| Quick start | `make demo` or [Quick Start](#quick-start) |
| Screenshot | [Portfolio Store](https://matthewpaver.github.io/MatthewPaver/store/) |
| Architecture | [Approach](#approach) |
| Tests | `make test` |
| Tech stack | `Python` `NumPy` `SciPy` `scikit-learn` `Jupyter` |

## Status

`Notebook project`

## Reviewer Pack

| Area | Details |
|:---|:---|
| What it solves | Uses swipe-style implicit feedback to rank unseen profiles and evaluate recommendations with temporal holdouts. |
| Screenshot | [Portfolio Store preview](https://matthewpaver.github.io/MatthewPaver/store/preview.html?app=recommender) |
| Run locally | `make demo` runs evaluation and a sample recommendation lookup using included data. |
| Tests | `make test` |
| Demo data | Included at `examples/sample_swipes.csv`; the larger anonymised dataset is tracked separately with Git LFS. |
| Architecture | CSV interactions -> positive implicit feedback -> temporal split -> sparse matrix -> SVD factors -> Top-K ranking |
| Limitations | Offline recommendation exercise, not a deployed recommender service with online feedback loops. |

## Reviewer Notes

- **Reproducible path:** the notebook is the narrative walkthrough; `recommender.py` gives a CLI route for repeatable checks.
- **ML signal:** the project treats swipes as implicit feedback and evaluates ranking quality with a temporal holdout.
- **Evaluation signal:** Hit Rate@K and MRR@K are exposed through the CLI rather than hidden inside notebook cells.
- **Known limit:** this is an offline recommendation exercise, not a deployed recommender service.

## Overview

Collaborative filtering recommendation system designed for swipe-based dating applications. Uses implicit feedback matrix factorisation (truncated SVD) to learn low-dimensional user and item embeddings from swipe data, then ranks unseen profiles by predicted affinity.

The notebook is the primary walkthrough. A lightweight CLI (`recommender.py`) provides a quick interface for dataset summary, model evaluation, and top-K lookups without opening Jupyter.

## Approach

![Recommendation system architecture](docs/assets/architecture.svg)

- **Implicit feedback** — treats positive swipes as signal; passes and unseen profiles are not assumed negative
- **Truncated SVD** — learns 32-dimension user and item factors from a sparse interaction matrix
- **Temporal hold-out** — evaluates on each user's most recent like, simulating real-world prediction
- **Metrics** — Hit Rate@K and MRR@K (Mean Reciprocal Rank)

## Quick Start

```bash
git clone https://github.com/MatthewPaver/dating-app-recommendation-system.git
cd dating-app-recommendation-system
make demo
```

The repo includes `examples/sample_swipes.csv` so the CLI can be tried without downloading the full Git LFS dataset.

### CLI

```bash
python recommender.py --csv examples/sample_swipes.csv summary
python recommender.py --csv examples/sample_swipes.csv evaluate --top-k 2
python recommender.py --csv examples/sample_swipes.csv recommend --user-id u1 --top-k 2
```

### Notebook

```bash
git lfs pull            # download swipes.csv if not already present
jupyter notebook data_scientist_exercise_anonymised.ipynb
```

Run all cells for the full analysis walkthrough: data preprocessing, model training, recommendation generation, and evaluation.

## Data Format

The system expects `swipes.csv` (tracked via Git LFS) with columns including `decidermemberid`, `othermemberid`, `timestamp`, `like`, gender, and signup metadata. Only positive swipes (`like = 1`) are used as training signal.

## Example Output

Dataset scale from the included anonymised CSV:

```text
Dataset summary
Users: 45588
Profiles: 77752
Positive interactions: 3413063
Date range: 2021-01-01 00:00:00 -> 2021-01-04 23:59:59
```

The CLI also exposes model evaluation and top-K lookup:

```text
python recommender.py evaluate --top-k 10
python recommender.py recommend --user-id <USER_ID> --top-k 10
```

## Repository Layout

```text
data_scientist_exercise_anonymised.ipynb   Main analysis notebook
recommender.py                             CLI for summary, evaluate, recommend
swipes.csv                                 Dataset (Git LFS)
requirements.txt                           Python dependencies
```

## Notes

- This is a technical exercise, not a deployed product. The notebook discusses production considerations (serving, cold-start, feedback loops) as design thinking, not implemented features.
- Git LFS is required for the dataset.
- CPU is sufficient for training.

## License

MIT. See [`LICENSE`](LICENSE).
