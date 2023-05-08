import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder


def retrieve(
    query: str,
    top_k: int = 5,
    model_name: str = "all-mpnet-base-v2",
    corpus_path: str = "embeddings/desc-embeddings-long.all-mpnet-base-v2.pkl",
):
    """Retrieve the most similar series in a corpus given a query"""

    # Load corpus embeddings
    with open(corpus_path, "rb") as f:
        corpus_embeddings = pickle.load(f)

    # Embed query
    model = SentenceTransformer(model_name)
    prompt_embedding = model.encode(query, convert_to_tensor=True)

    # Search for results
    results = util.semantic_search(prompt_embedding, corpus_embeddings, top_k=top_k)[0]
    results = pd.DataFrame(results, columns=["corpus_id", "score"])

    return results


def rerank(
    query: str,
    retrieved: pd.DataFrame,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """Re-rank the retrieved series using a cross encoder"""

    # Create pairs of query and descriptions
    inp = [[query, desc] for desc in retrieved["desc"]]

    # Get scores for each pair
    cross_encoder = CrossEncoder(model_name)
    cross_scores = cross_encoder.predict(inp)
    retrieved["cross-score"] = cross_scores

    # Get top-k after re-ranking
    results = retrieved.sort_values("cross-score", ascending=False).iloc[:top_k]

    return results


if __name__ == "__main__":
    data = pd.read_csv("data/cleaned-long.csv")
    q = "a series about people battling each other in cooking competitions"
    results = retrieve(q, top_k=50)

    idxs = results["corpus_id"].tolist()
    descs = data.iloc[idxs].input.tolist()
    results["desc"] = descs
    print(results)

    reranked = rerank(q, results, top_k=5)
    print(reranked)
