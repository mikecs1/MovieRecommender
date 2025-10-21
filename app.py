import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests

# --- Load data & FAISS index ---
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    # Handle missing release_date
    df["release_date"] = df["release_date"].fillna("Unknown")

    # Create textual representation
    df["textual_representation"] = df.apply(
        lambda row: f"Title: {row['title']}\nDescription: {row['overview']}\nYear: {str(row['release_date'])[:4]}",
        axis=1
    )
    return df

@st.cache_resource
def load_index():
    index = faiss.read_index("tmdb_index.faiss")
    return index

# --- Load precomputed embeddings ---
embeddings = np.load("tmdb_embeddings.npy")  # shape: [num_movies, dim]


df = load_data()
index = load_index()

# --- UI ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("Movie Recommender")
st.write("Find movies similar to your favorite titles using AI + FAISS.")

movie_title = st.selectbox(
    "Choose a movie to get recommendations:",
    sorted(df["title"].dropna().unique())
)

# Helper function to fetch poster
def get_poster(title):
    api_key = st.secrets.get("TMDB_API_KEY", None)
    if not api_key:
        return None
    try:
        # Search for the movie
        res = requests.get(
            f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
        ).json()
        if res["results"]:
            poster_path = res["results"][0].get("poster_path")
            if poster_path:
                # Full image URL
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        st.write(f"⚠️ Error fetching poster for {title}: {e}")
        return None


if movie_title:
    movie_row = df[df["title"] == movie_title].iloc[0]
    release_year = str(movie_row["release_date"])[:4]
    st.subheader(f"🎥 {movie_row['title']} ({release_year})")

    # Show poster if available
    poster = get_poster(movie_row["title"])
    if poster:
        st.image(poster, width=250)

    st.write(movie_row["overview"])  # use TMDB overview instead of description

    # --- Create embedding ---
    # --- Use precomputed embedding ---
    movie_idx = df[df["title"] == movie_title].index[0]
    embedding = embeddings[movie_idx:movie_idx+1]  # shape (1, dim)

    # --- Search similar movies ---
    D, I = index.search(embedding, k=6)

    st.subheader(" Recommended Similar Movies ")
    for i in I[0][1:]:
        rec = df.iloc[i]
        rec_poster = get_poster(rec["title"])  # TMDB poster function

        col1, col2 = st.columns([1, 3])
        with col1:
            if rec_poster:
                st.image(rec_poster, width=120)
        with col2:
            st.markdown(f"**{rec['title']} ({str(rec['release_date'])[:4]})**")
            st.write(rec["overview"])
        st.divider()
