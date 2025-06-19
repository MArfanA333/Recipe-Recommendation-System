from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

def select_random_top_rated_recipe(authorId, reviews_df, recipes_df, top_n=10):
    # Filter and get user reviews sorted by rating
    user_reviews = reviews_df[reviews_df['AuthorId'] == authorId].sort_values(by='Rating', ascending=False)
    top_reviews = user_reviews.head(top_n)
    
    # Get the corresponding recipes with valid descriptions
    top_recipes = recipes_df[recipes_df['RecipeId'].isin(top_reviews['RecipeId'])].dropna(subset=['Description'])

    # Randomly select one
    if top_recipes.empty:
        raise ValueError("No valid top-rated recipes with descriptions found for this user.")
    
    return top_recipes.sample(1).iloc[0]

def tf_idf_single_recipe(authorId, reviews_df, recipes_df):
    # Select one top-rated recipe
    chosen_recipe = select_random_top_rated_recipe(authorId, reviews_df, recipes_df)

    # Candidate recipes: not reviewed by the user and have descriptions
    user_reviewed_ids = reviews_df[reviews_df['AuthorId'] == authorId]['RecipeId'].unique()
    candidate_recipes = recipes_df[~recipes_df['RecipeId'].isin(user_reviewed_ids)].dropna(subset=['Description']).copy()

    # TF-IDF vectorization
    corpus = [chosen_recipe['Description']] + candidate_recipes['Description'].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vector = tfidf_matrix[0].reshape(1, -1)
    candidate_matrix = tfidf_matrix[1:]

    # Compute similarities
    similarities = cosine_similarity(candidate_matrix, query_vector)

    # Attach similarities to candidates
    candidate_recipes['Similarity'] = similarities.flatten()
    sorted_recipes = candidate_recipes.sort_values(by='Similarity', ascending=False)

    return sorted_recipes, chosen_recipe[['RecipeId', 'Name', 'Description']]
