# 🍽️ Recipe Recommendation System

A hybrid recommendation engine built on a large-scale recipe dataset, integrating collaborative filtering, content-based techniques, and non-personalized scoring to help users discover recipes tailored to their preferences.

---

## 📄 Project Overview

This system is designed for intelligent recipe discovery on large platforms like Food.com. It addresses challenges like cold-start users, sparse ratings, and scalable recommendation delivery by combining:

- **Collaborative Filtering (SVD)** for personalized rating predictions
- **Content-Based Filtering (TF-IDF)** for textual profile matching
- **Jaccard Similarity** for cold-start users
- **Bayesian Scoring** for globally fair popularity rankings

These methods are fused into a hybrid recommendation engine that offers accuracy, flexibility, and adaptability for new and returning users.

---


## 📁 Folder Structure

```plaintext
📁 Recipe-Recommendation-System/
├── README.md # This file
├── jaccard_bayesian_calc.py # Defines functions to calculate jaccard score and bayesian
├── tf_idf_all_w_reason # Defines function to calculate tf-idf score and recommend recipes with reason for a user
└── tf_idf_single # Defines a function to calculate tf-idf score for a randomly selected recipe and recommend for a user
````
---

## 🧠 Dataset

**Source:** [Food.com Recipes and Reviews - Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)  
- 500,000+ recipes from 312 categories  
- 1.4 million user reviews from over 270,000 users  
- Rich metadata including ingredients, nutrition, keywords, cooking time, and more  

**Preprocessing Highlights:**
- Removed irrelevant or sparse columns (e.g., `Images`, `RecipeYield`)
- Converted ISO 8601 time fields into minutes
- Imputed missing values in `RecipeServings`
- Parsed and cleaned keyword fields (originally in R-style lists)
- Sampled 250,000 recipes for manageable processing

---

## 🛠️ Methodology

### 🔹 Cold Start — Jaccard Similarity
New users select a set of keywords (e.g., "vegetarian", "spicy") on first login. These are compared to recipe keywords using Jaccard similarity, and the most overlapping recipes are recommended — enabling personalized suggestions without prior user history.

### 🔹 Collaborative Filtering — SVD
SVD is applied on the user–recipe matrix to uncover latent features. The model centers user ratings, performs matrix decomposition, then reconstructs the matrix and adds back user means to get predicted scores.

### 🔹 Content-Based Filtering — TF-IDF + Cosine Similarity
We build user preference profiles from the recipe `Description` fields of previously reviewed items. TF-IDF vectors are generated and compared using cosine similarity to recommend unseen recipes that match the user's textual taste profile.

### 🔹 Bayesian Scoring
Used for non-personalized but trustworthy global ranking. Scores are calculated using:

BayesianScore = (v / (v + m)) \* R + (m / (v + m)) \* C

Where:
- `R` = average recipe rating
- `v` = number of votes
- `C` = global average rating
- `m` = minimum votes threshold (empirically chosen)

---

## 📊 Results

### ✅ Jaccard Similarity
Successfully recommended highly relevant recipes for new users based on keyword overlap. Best suited for onboarding and preference surveys.

### ✅ Bayesian Scoring
Produced stable, fair rankings of recipes by combining quality and popularity — useful for trending and featured sections.

### ✅ SVD
Predicted user ratings effectively but suffered due to rating bias (most users rated 4–5 stars). Still useful for personalized, rating-based recommendations.

### ✅ TF-IDF
Enabled meaningful suggestions even for users without numeric ratings. Used F1@K with K=50,000 to evaluate performance:
- Precision: 0.0031
- Recall: 0.1792
- F1-score: 0.0062

---

## 💡 Discussion

- **Jaccard** is fast and ideal for new users, but dependent on keyword quality.
- **Bayesian** ranks globally popular content with fairness, but lacks personalization.
- **SVD** is strong in personalizing recommendations from rating data, but slow and opaque.
- **TF-IDF** is great for unrated data but prone to textual repetition.

Together, these methods support a **hybrid strategy** that ensures robustness across user scenarios.

---

## ⚠️ Limitations

- **Rating skew**: Over 88% of ratings are 4–5, limiting SVD's effectiveness.
- **Opacity**: SVD recommendations are not explainable to users.
- **TF-IDF limitations**: Ignores semantic similarity and context.
- **Scalability**: High computation time for real-time inference.

---

## 🔮 Future Work

- Integrate deep learning models (e.g., BERT) for better content understanding
- Use contextual signals (e.g., time, trends) for dynamic recommendations
- Optimize runtime with approximate nearest neighbor methods
- Enhance transparency with explainable AI techniques

---

## 👥 Team

- **Mohammed Arfan Ameen**
- **Mustafa Ashraf**  
- **Sean Timothy Arquero**  

---
