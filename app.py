import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from inference import search

DATA_PATH = "data/cleaned-long.csv"


def get_data(results: pd.DataFrame, data: pd.DataFrame):
    """Given the corpus indices of the top-k series get the required data for the UI"""
    titles, scores, covers, urls = [], [], [], []
    for idx, score in zip(results.corpus_id.tolist(), results.score.tolist()):
        titles.append(data.iloc[idx].romaji)
        scores.append(score)
        covers.append(data.iloc[idx].cover)
        urls.append(data.iloc[idx].url)

    return titles, scores, covers, urls


# Load corpus dataframe
data = pd.read_csv(DATA_PATH)

# Input UI
st.title("Manga Semantic Search")
query = st.text_input(
    "Enter a description of the manga you are searching for:",
    value="",
)
embeddings_path = st.selectbox("Embeddings Corpus", os.listdir("embeddings"))
top_k = st.number_input(
    "Number of results", value=5, min_value=1, max_value=100, step=1
)

model_name = str(embeddings_path).split(".")[-2]
embeddings_path = os.path.join("embeddings", str(embeddings_path))

# Output UI
if st.button("Search"):
    # Check that query is not empty
    if not query:
        st.write("Please enter a query")

    else:
        # Find most similar series
        results = search(
            query, top_k=int(top_k), model_name=model_name, corpus_path=embeddings_path
        )
        titles, scores, covers, urls = get_data(results, data)

        # Display results
        for title, score, cover, url in zip(titles, scores, covers, urls):
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"""
                            ## [{title}]({url})
                            Score: {score:.2f}
                        """
                    )
                with col2:
                    response = requests.get(cover)
                    img = BytesIO(response.content)
                    st.image(img, width=200)
