import pandas as pd
import numpy as np
import faiss
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # for better progress bar

# Load TMDB data
df = pd.read_csv("tmdb_5000_movies.csv")

def create_textual_representation(row):
    return f"""Title: {row['title']}
    Description: {row['overview']}
    Year: {str(row['release_date'])[:4]}
    """

index = faiss.read_index('tmdb_index.faiss')

df['textual_representation'] = df.apply(create_textual_representation, axis=1)

# search_movies = df[df.title.str.contains('Shutter')]
# print("\nFound these movies: ")
# print(search_movies[['title', 'release_year', 'description']])

fav_movie = df.sample(1).iloc[0]
text = create_textual_representation(fav_movie)
# print(fav_movie[['title', 'release_year', 'description']])

res = requests.post('http://localhost:11434/api/embeddings', json={
    'model': 'all-minilm:l6',
    'prompt': text
})

embedding = np.array([res.json()['embedding']], dtype='float32')

D, I = index.search(embedding, k=10)

# print(I)
best_matches = np.array(df['textual_representation'])[I.flatten()]

for match in best_matches:
    print('Next movie : ')
    print(match)
    

# Create a session to reuse connections
# session = requests.Session()

# def get_embedding(text):
#     """Get one embedding using the faster model"""
#     try:
#         res = session.post('http://localhost:11434/api/embeddings',
#                         json={
#                             'model': 'all-minilm:l6',
#                             'prompt': text
#                         })
#         return np.array(res.json()['embedding'], dtype='float32')
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# print("Creating text representations...")
# texts = df.apply(create_textual_representation, axis=1)
# print(f"Processing {len(texts)} items...")

# dim = 384  
# index = faiss.IndexFlatL2(dim)

# embeddings = []
# with ThreadPoolExecutor(max_workers=6) as executor:  
#     # Use tqdm for progress bar
#     futures = list(tqdm(
#         executor.map(get_embedding, texts),
#         total=len(texts),
#         desc="Getting embeddings"
#     ))
    
#     # Filter out any failed embeddings
#     embeddings = [emb for emb in futures if emb is not None]

# # Stack into array and add to index
# X = np.vstack(embeddings)
# index.add(X)

# # Save the index
# faiss.write_index(index, 'netflix_index.faiss')
# print(f"Done! Processed {len(embeddings)} items")