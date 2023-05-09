import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer


def build_corpus(
    data_path: str,
    model_name: str = "all-mpnet-base-v2",
    output_path: str = "desc-embeddings.pkl",
):
    """Embed all series descriptions and save results to file"""

    # Load descriptions
    data = pd.read_csv(data_path)
    descs = data.input.tolist()

    # Load model
    model = SentenceTransformer(model_name)
    print(model[0])

    # Embed descriptions
    corpus_embeddings = model.encode(descs, show_progress_bar=True)

    # Save embeddings and data
    combined = {"data": data, "embeddings": corpus_embeddings}
    with open(output_path, "wb") as f:
        pickle.dump(combined, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to data csv"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="desc-embeddings.pkl",
        type=str,
        help="Output path of pickled corpus embeddings",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="all-mpnet-base-v2",
        type=str,
        help="Name of embedding model. List of models can be found at https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/",
    )
    args = parser.parse_args()

    build_corpus(args.input, args.model, args.output)

    print(f"Finished building the embedding corpus. Results written to {args.output}")
