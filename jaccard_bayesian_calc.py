import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

# Load reviews dataset
reviews_df = pd.read_csv('C:\Users\User\Desktop\IRS Python FIles\Project\reviews.csv')

# --- JACCARD SIMILARITY ---

def compute_jaccard_similarity(df):
    user_items = df.groupby('user_id')['recipe_id'].apply(set)
    users = user_items.index.tolist()
    
    similarities = []

    for u1, u2 in tqdm(combinations(users, 2)):
        set1, set2 = user_items[u1], user_items[u2]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union > 0:
            score = intersection / union
            similarities.append({'user_1': u1, 'user_2': u2, 'jaccard_similarity': score})

    sim_df = pd.DataFrame(similarities)
    sim_df.to_csv('jaccard_similarity.csv', index=False)
    print("Saved Jaccard similarity to 'jaccard_similarity.csv'")

# --- BAYESIAN AVERAGE SCORING ---

def compute_bayesian_score(df):
    recipe_group = df.groupby('recipe_id')['rating']
    recipe_counts = recipe_group.count()
    recipe_means = recipe_group.mean()

    C = recipe_counts.mean()  # average number of ratings
    m = recipe_means.mean()   # global average rating

    bayesian_scores = ((recipe_counts * recipe_means) + (C * m)) / (recipe_counts + C)

    bayesian_df = bayesian_scores.reset_index()
    bayesian_df.columns = ['recipe_id', 'bayesian_score']
    bayesian_df.to_csv('bayesian_scores.csv', index=False)
    print("Saved Bayesian scores to 'bayesian_scores.csv'")

# Run computations
#compute_jaccard_similarity(reviews_df)
compute_bayesian_score(reviews_df)