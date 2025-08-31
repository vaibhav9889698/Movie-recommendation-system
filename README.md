# Movie-recommendation-system

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
movies = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'The Dark Knight', 'Inception', 'Interstellar'],
    'genres': ['Action Sci-Fi', 'Action Thriller', 'Action Crime Drama', 'Action Sci-Fi Thriller', 'Sci-Fi Drama Adventure']
}

df = pd.DataFrame(movies)

# Convert genres into vectors (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_content(movie_title):
    idx = df.index[df['title'] == movie_title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example
print("Recommendations for 'Inception':")
print(recommend_content('Inception'))

