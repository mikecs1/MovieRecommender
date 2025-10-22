# MovieRecommender

A Python-based movie recommendation system that helps users discover new films based on their preferences and viewing history.

## Overview

MovieRecommender is a project that uses machine learning and data analysis techniques to provide personalized movie recommendations to users. It supports classic approaches (content-based and collaborative filtering) and can be extended into hybrid models.

## Features

- Personalized recommendations based on user preferences and viewing history
- Multiple algorithms: content-based, item-based/user-based collaborative filtering
- Pluggable similarity metrics (cosine, Pearson, etc.)
- Data analysis utilities for exploring movie datasets
- Pythonic API and optional CLI for easy usage
- Reproducible training and evaluation workflows


## Installation

1) Clone the repository:
```bash
git clone https://github.com/mikecs1/MovieRecommender.git
cd MovieRecommender
```

2) Create and activate a virtual environment (recommended):
```bash
# with venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3) Install dependencies:
```bash
pip install -U pip
pip install -r requirements.txt
```

Optional (dev tools):
```bash
pip install -r requirements-dev.txt
```

## Algorithms

- Content-Based Filtering
  - Uses item features (e.g., genres, embeddings) to compute similarity
- Collaborative Filtering
  - User-User and Item-Item variants with cosine or Pearson similarity
- Hybrid
  - Weighted/blended scores from multiple models

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- MovieLens datasets by GroupLens Research
- scikit-learn, pandas, numpy, scipy, and the Python open-source community
