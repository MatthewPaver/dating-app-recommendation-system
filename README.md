# Dating App Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

**User-User Collaborative Filtering for Swipe-Based Dating Apps**

*A production-ready recommendation system using implicit feedback matrix factorisation*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Approach](#-approach)
- [Model Details](#-model-details)
- [Getting Started](#-getting-started)
- [Data Format](#-data-format)
- [Results](#-results)
- [Production Considerations](#-production-considerations)
- [Dependencies](#-dependencies)
- [Repository Structure](#-repository-structure)

---

## 🎯 Overview

This project implements a collaborative filtering recommendation system designed for swipe-based dating applications. The system uses implicit feedback matrix factorisation to identify which profiles to show each user, maximising the likelihood of a match.

### Problem Statement

In a dating app, users swipe on profiles (like or pass). The challenge is to recommend profiles that each user is most likely to like, based on their historical swipe patterns and the patterns of similar users.

### Solution

The system uses **truncated SVD (Singular Value Decomposition)** on implicit feedback data (positive swipes/likes) to learn low-dimensional embeddings of users and profiles. These embeddings capture latent preferences and enable personalised recommendations.

---

## ✨ Key Features

- **Implicit Feedback Processing** — Treats positive swipes (`like = 1`) as implicit positive feedback
- **Matrix Factorisation** — Uses truncated SVD to learn low-dimensional user and item embeddings
- **Temporal Validation** — Holds out each user's most recent like for evaluation
- **Efficient Computation** — Sparse matrix implementation for scalable processing
- **Top-K Recommendations** — Generates ranked recommendations with filtering of already-seen profiles
- **Production-Ready** — Includes deployment considerations and architecture guidance

---

## 🔬 Approach

### Implicit Feedback Matrix Factorisation

Unlike explicit ratings (e.g., 1–5 stars), dating apps only provide implicit feedback:
- **Positive signal**: User swiped right (liked)
- **No signal**: User swiped left (passed) or hasn't seen the profile

The system treats positive swipes as implicit positive feedback and uses matrix factorisation to learn latent user and item factors.

### Methodology

1. **Data Preprocessing**
   - Construct user-item interaction matrix from swipe data
   - Use sparse matrices for memory efficiency
   - Handle temporal ordering for validation

2. **Model Training**
   - Apply truncated SVD to learn low-dimensional embeddings
   - Learn user and item factors that capture latent preferences
   - Use 32 latent dimensions (configurable)

3. **Recommendation Generation**
   - Compute predicted scores for all unseen profiles
   - Rank profiles by predicted score
   - Filter out already-seen profiles
   - Return top-K recommendations

4. **Evaluation**
   - Temporal hold-out validation (most recent like per user)
   - Hit Rate@K: Percentage of users with at least one relevant recommendation in top-K
   - MRR@K (Mean Reciprocal Rank): Average reciprocal rank of first relevant item

---

## 🧮 Model Details

### Algorithm

- **Method**: Truncated SVD (Singular Value Decomposition)
- **Latent Dimensions**: 32 (configurable)
- **Objective**: Minimise reconstruction error of user-item interaction matrix

### Evaluation Metrics

- **Hit Rate@10**: Percentage of users with at least one relevant recommendation in top 10
- **MRR@10**: Mean Reciprocal Rank of first relevant item in top 10

### Validation Strategy

- **Temporal Split**: Holds out each user's most recent like for evaluation
- **Train Set**: All historical likes except the most recent
- **Test Set**: Most recent like per user

This approach simulates real-world scenarios where recommendations are made based on historical data.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Git LFS (for downloading the data file)
- Jupyter Notebook

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MatthewPaver/dating-app-recommendation-system.git
   cd dating-app-recommendation-system
   ```

2. **Install Git LFS** (if not already installed):
   ```bash
   # macOS
   brew install git-lfs
   git lfs install
   
   # Linux
   sudo apt-get install git-lfs
   git lfs install
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open the notebook:**
   ```bash
   jupyter notebook data_scientist_exercise_anonymised.ipynb
   ```

5. **Run all cells** to execute the analysis

### Quick Start

The notebook will automatically download `swipes.csv` via Git LFS when you clone the repository. Simply run all cells to:
- Load and preprocess the data
- Train the recommendation model
- Generate recommendations
- Evaluate performance

---

## 📊 Data Format

The system expects a CSV file (`swipes.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| `decidermemberid` | User making the swipe |
| `othermemberid` | Profile being swiped on |
| `timestamp` | UTC timestamp of the swipe |
| `like` | 1 for like, 0 for pass |
| `decidergender` / `othergender` | Gender information |
| `deciderdobyear` / `otherdobyear` | Year of birth |
| `decidersignuptimestamp` / `othersignuptimestamp` | Signup timestamps |

### Data Characteristics

- **Implicit Feedback**: Only positive signals (likes) are used
- **Sparse Matrix**: Most user-item pairs have no interaction
- **Temporal Ordering**: Swipes are ordered by timestamp

---

## 📈 Results

The model generates personalised recommendations for users with sufficient interaction history. Key findings:

- **Hit Rate@10**: Measures the percentage of users for whom at least one relevant profile appears in the top 10 recommendations
- **MRR@10**: Measures the average quality of ranking, with higher values indicating better performance

The evaluation demonstrates that the model successfully:
- Learns meaningful user and item embeddings
- Generates personalised recommendations
- Performs well on temporal hold-out validation

### Example Recommendations

The notebook includes example recommendations for test cases, showing how the model identifies profiles that users are likely to find appealing based on their historical preferences.

---

## 🏭 Production Considerations

The notebook includes a comprehensive section on deploying this system in production, covering:

### Data Pipeline

- **Streaming Data**: Real-time ingestion of swipe events
- **Batch Processing**: Periodic retraining on updated data
- **Data Quality**: Validation and monitoring of input data

### Serving Architecture

- **Model Serving**: Efficient serving of recommendations at scale
- **Caching**: Caching user embeddings and top-K recommendations
- **Load Balancing**: Distributing requests across multiple servers

### Cold-Start Handling

- **New Users**: Strategies for users with no interaction history
- **New Profiles**: Strategies for newly added profiles
- **Hybrid Approaches**: Combining collaborative filtering with content-based methods

### Business Rules & Safety

- **Filtering**: Excluding inappropriate or blocked profiles
- **Diversity**: Ensuring diverse recommendations
- **Fairness**: Avoiding bias in recommendations

### Feedback Loops

- **Continuous Learning**: Updating models based on new interactions
- **A/B Testing**: Testing different recommendation strategies
- **Monitoring**: Tracking recommendation quality and user engagement

---

## 📚 Dependencies

### Core Libraries

- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computing
- **scipy** — Sparse matrix operations and scientific computing
- **jupyter** — Interactive notebook environment

### Installation

```bash
pip install pandas numpy scipy jupyter
```

Or install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
dating-app-recommendation-system/
├── data_scientist_exercise_anonymised.ipynb  # Main analysis notebook
├── swipes.csv                                # Dataset (Git LFS)
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## 💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

</div>

---

## 📝 License

This is a technical exercise/prototype. See the notebook for full details.

---

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome! Feel free to open an issue or submit a pull request.

---

## 📖 Additional Resources

- [Matrix Factorisation for Recommender Systems](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
- [Implicit Feedback in Recommender Systems](https://www.benfrederickson.com/matrix-factorization/)
- [Evaluating Recommender Systems](https://grouplens.org/blog/evaluating-recommender-systems/)

---

<div align="center">

**Built for understanding collaborative filtering and recommendation systems**

[⬆ Back to Top](#dating-app-recommendation-system)

</div>
