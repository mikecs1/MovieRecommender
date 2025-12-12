import pandas as pd
import numpy as np
import faiss

# Load TMDB data
df = pd.read_csv("tmdb_5000_movies.csv")

def create_textual_representation(row):
    return f"""Title: {row['title']}
    Description: {row['overview']}
    Year: {str(row['release_date'])[:4]}
    """

index = faiss.read_index('tmdb_index.faiss')

df['textual_representation'] = df.apply(create_textual_representation, axis=1)

all_embeddings = np.load('tmdb_embeddings.npy')

random_index = df.sample(1).index[0]
fav_movie = df.loc[random_index]

embedding = np.array([all_embeddings[random_index]], dtype='float32')

D, I = index.search(embedding, k=10)

print(f"Selected Movie: {fav_movie['title']}")
best_matches = np.array(df['textual_representation'])[I.flatten()]

for match in best_matches:
    print('---')
    print(match)