# import pandas as pd
# import numpy as np
# import faiss
# import requests
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# df = pd.read_csv("tmdb_5000_movies.csv")

# def create_textual_representation(row):
#     return f"""Title: {row['title']}
# Description: {row['overview']}
# Year: {str(row['release_date'])[:4]}
# """

# df["textual_representation"] = df.apply(create_textual_representation, axis=1)

# session = requests.Session()

# def get_embedding(text):
#     try:
#         res = session.post("http://localhost:11434/api/embeddings", json={
#             "model": "all-minilm:l6",
#             "prompt": text
#         })
#         return np.array(res.json()["embedding"], dtype="float32")
#     except:
#         return None

# embeddings = []
# with ThreadPoolExecutor(max_workers=6) as executor:
#     results = list(tqdm(executor.map(get_embedding, df["textual_representation"]), total=len(df)))
#     embeddings = [r for r in results if r is not None]

# X = np.vstack(embeddings)
# index = faiss.IndexFlatL2(X.shape[1])
# index.add(X)
# faiss.write_index(index, "tmdb_index.faiss")
