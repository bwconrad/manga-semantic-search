import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from inference import retrieve, rerank

DATA_PATH = "data/cleaned-long.csv"


def get_data(results: pd.DataFrame, data: pd.DataFrame, reranked=False):
    """Given the corpus indices of the top-k series get the required data for the UI"""
    if reranked:
        scores_list = results["cross-score"].tolist()
    else:
        scores_list = results.score.tolist()

    titles, scores, covers, urls = [], [], [], []
    for idx, score in zip(results.corpus_id.tolist(), scores_list):
        titles.append(data.iloc[idx].romaji)
        scores.append(score)
        covers.append(data.iloc[idx].cover)
        urls.append(data.iloc[idx].url)

    return titles, scores, covers, urls


def add_descriptions_to_results(results: pd.DataFrame):
    """Add the corresponding description to the retrieval results"""
    idxs = results["corpus_id"].tolist()
    descs = data.iloc[idxs].input.tolist()
    results["desc"] = descs
    return results


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
do_rerank = st.checkbox("Re-Rank", value=True)
k_retrieve = None
if do_rerank:
    k_retrieve = st.number_input(
        "Number of initialy retrieved series",
        value=50,
        min_value=1,
        max_value=500,
        step=1,
    )


# Convert UI values into the correct function argument values
model_name = str(embeddings_path).split(".")[-2]
embeddings_path = os.path.join("embeddings", str(embeddings_path))

# Output UI
if st.button("Search"):
    if not k_retrieve:
        k_retrieve = top_k

    # Check that query is not empty
    if not query:
        st.write("Please enter a query")
    # Check that top_k is not > retrieve_k
    elif top_k > k_retrieve:
        st.write(
            "'Number of results' should be less than or equal to 'Number of number of initialy retrieved series'"
        )
    else:
        # Retrieve most similar series
        results = retrieve(
            query,
            corpus_path=embeddings_path,
            model_name=model_name,
            top_k=int(k_retrieve),
        )
        # Re-rank the retrieved series
        if do_rerank:
            results = add_descriptions_to_results(results)
            results = rerank(query, results, top_k=int(top_k))

        # Display results
        titles, scores, covers, urls = get_data(results, data, do_rerank)
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
