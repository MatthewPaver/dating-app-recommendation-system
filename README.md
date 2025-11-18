# Dating App Recommendation System

A user-user recommendation system for swipe-based dating apps, built using implicit feedback matrix factorisation. This system identifies which profiles to show each user to maximise the likelihood of a match.

## Overview

This project implements a collaborative filtering recommendation system that identifies which profiles to show each user to maximise the likelihood of a match. The solution uses truncated SVD (Singular Value Decomposition) on implicit feedback data (positive swipes/likes).

## Approach

- **Implicit Feedback**: Treats positive swipes (`like = 1`) as implicit positive feedback
- **Matrix Factorisation**: Uses truncated SVD to learn low-dimensional user and item embeddings
- **Evaluation**: Hit Rate@K and MRR@K metrics with temporal hold-out validation
- **Production-Ready**: Includes considerations for deployment at scale

## Key Features

- Temporal train/validation split (holds out each user's most recent like)
- Sparse matrix implementation for efficient computation
- Top-K recommendation generation with filtering of seen profiles
- Example recommendations for test cases
- Production deployment considerations

## Requirements

See `requirements.txt` for dependencies. Main libraries:
- pandas
- numpy
- scipy
- jupyter

## Usage

1. Clone the repository (Git LFS will automatically download `swipes.csv`)
2. Open `data_scientist_exercise_anonymised.ipynb` in Jupyter
3. Run all cells

**Note**: This repository uses Git LFS for the data file. If you don't have Git LFS installed, install it first:
```bash
brew install git-lfs  # macOS
git lfs install
```

The notebook expects a CSV file with the following columns:
- `decidermemberid`: User making the swipe
- `othermemberid`: Profile being swiped on
- `timestamp`: UTC timestamp of the swipe
- `like`: 1 for like, 0 for pass
- `decidergender` / `othergender`: Gender information
- `deciderdobyear` / `otherdobyear`: Year of birth
- `decidersignuptimestamp` / `othersignuptimestamp`: Signup timestamps

## Model Details

- **Algorithm**: Truncated SVD with 32 latent dimensions
- **Evaluation Metrics**: Hit Rate@10, MRR@10
- **Validation**: Temporal split (most recent like per user held out)

## Results

The model generates personalised recommendations for users with sufficient interaction history. Evaluation metrics measure ranking quality on held-out validation data.

## Production Considerations

The notebook includes a section on how this system would work in production, covering:
- Data streaming and training pipelines
- Serving architecture
- Cold-start handling
- Business rules and safety controls
- Feedback loops and continuous improvement

## License

This is a technical exercise/prototype. See the notebook for full details.

