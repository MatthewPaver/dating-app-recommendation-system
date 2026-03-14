# Dating App Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

**User-user collaborative filtering for swipe-based dating apps**

*Notebook project with a lightweight CLI for evaluation and recommendation lookup*

</div>

---

## Status

`Notebook project`

## Overview

Collaborative filtering recommendation system designed for swipe-based dating applications. Uses implicit feedback matrix factorisation (truncated SVD) to learn low-dimensional user and item embeddings from swipe data, then ranks unseen profiles by predicted affinity.

The notebook is the primary walkthrough. A lightweight CLI (`recommender.py`) provides a quick interface for dataset summary, model evaluation, and top-K lookups without opening Jupyter.

## Approach

- **Implicit feedback** — treats positive swipes as signal; passes and unseen profiles are not assumed negative
- **Truncated SVD** — learns 32-dimension user and item factors from a sparse interaction matrix
- **Temporal hold-out** — evaluates on each user's most recent like, simulating real-world prediction
- **Metrics** — Hit Rate@K and MRR@K (Mean Reciprocal Rank)

## Quick Start

```bash
git clone https://github.com/MatthewPaver/dating-app-recommendation-system.git
cd dating-app-recommendation-system
pip install -r requirements.txt
git lfs pull            # download swipes.csv if not already present
```

### CLI

```bash
python recommender.py summary
python recommender.py evaluate --top-k 10
python recommender.py recommend --user-id <USER_ID> --top-k 10
```

### Notebook

```bash
jupyter notebook data_scientist_exercise_anonymised.ipynb
```

Run all cells for the full analysis walkthrough: data preprocessing, model training, recommendation generation, and evaluation.

## Data Format

The system expects `swipes.csv` (tracked via Git LFS) with columns including `decidermemberid`, `othermemberid`, `timestamp`, `like`, gender, and signup metadata. Only positive swipes (`like = 1`) are used as training signal.

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
