import pickle
import torch

import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder


def retrieve(
    query: str,
    corpus_embeddings: torch.Tensor,
    top_k: int = 5,
    model_name: str = "all-mpnet-base-v2",
):
    """Retrieve the most similar series in a corpus given a query"""

    # Embed query
    model = SentenceTransformer(model_name)
    prompt_embedding = model.encode(query, convert_to_tensor=True)

    # Find most similar
    results = util.semantic_search(prompt_embedding, corpus_embeddings, top_k=top_k)[0]
    results = pd.DataFrame(results, columns=["corpus_id", "score"])

    return results


def rerank(
    query: str,
    retrieved: pd.DataFrame,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """Re-rank the retrieved series"""

    # Create pairs of query and descriptions
    inp = [[query, desc] for desc in retrieved["desc"]]

    # Get scores for each pair
    cross_encoder = CrossEncoder(model_name)
    cross_scores = cross_encoder.predict(inp)
    retrieved["cross-score"] = cross_scores

    # Keep top-k after re-ranking
    results = retrieved.sort_values("cross-score", ascending=False).iloc[:top_k]

    return results


if __name__ == "__main__":
    with open("embeddings/desc-embeddings.all-mpnet-base-v2.pkl", "rb") as f:
        data, corpus_embeddings = pickle.load(f).values()

    q = "a series about people battling each other in cooking competitions"
    results = retrieve(q, corpus_embeddings, top_k=50)

    idxs = results["corpus_id"].tolist()
    descs = data.iloc[idxs].input.tolist()
    results["desc"] = descs
    print(results)

    reranked = rerank(q, results, top_k=5)
    print(reranked)
