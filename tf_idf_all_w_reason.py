from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def tf_idf(authorId, reviews_df, recipes_df):
    # Get the recipes reviewed by the user
    user_reviews = reviews_df[reviews_df['AuthorId'] == authorId]
    user_recipe_ids = user_reviews['RecipeId'].unique()
    user_recipes = recipes_df[recipes_df['RecipeId'].isin(user_recipe_ids)].dropna(subset=['Description']).copy()
    print('Got user recipes')

    # Get candidate recipes (not reviewed by the user)
    candidate_recipes = recipes_df[~recipes_df['RecipeId'].isin(user_recipe_ids)].dropna(subset=['Description']).copy()
    print('Preparing candidate recipes')

    # Combine descriptions for TF-IDF vectorization
    all_descriptions = user_recipes['Description'].tolist() + candidate_recipes['Description'].tolist()

    print('Starting TF-IDF vectorization')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    print('Vectorization done')

    # Separate the matrices
    user_matrix = tfidf_matrix[:len(user_recipes)]
    candidate_matrix = tfidf_matrix[len(user_recipes):]

    print('Computing similarity matrix')
    similarity_matrix = cosine_similarity(candidate_matrix, user_matrix)

    # Find max similarity and matching recipe index for each candidate
    max_similarities = similarity_matrix.max(axis=1)
    best_match_indices = similarity_matrix.argmax(axis=1)

    # Get the names of the most similar user-reviewed recipes
    best_matches = user_recipes.iloc[best_match_indices].reset_index(drop=True)
    candidate_recipes = candidate_recipes.reset_index(drop=True)

    # Add similarity scores and matched recipe names
    candidate_recipes['Similarity'] = max_similarities
    candidate_recipes['MatchedRecipe'] = best_matches['Name']

    # Sort recommendations
    sorted_recipes = candidate_recipes.sort_values(by='Similarity', ascending=False)

    return sorted_recipes[['RecipeId', 'Name', 'Similarity', 'MatchedRecipe']]
