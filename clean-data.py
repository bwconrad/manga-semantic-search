import re
from argparse import ArgumentParser
from ast import literal_eval

import numpy as np
import pandas as pd


def clean_description(row: pd.Series, tag_thresh: float = 0.5):
    """Entry description"""
    desc = row.description
    if not isinstance(desc, str):
        desc = ""
    else:
        # Remove <br>
        desc = desc.replace("<br>", "")

        # Find the (Source: ...) and only keep whats before
        pattern = r"\(Source: .*\)"
        match = re.search(pattern, desc)
        if match:
            desc = desc[: match.start()]

        # If no (Source: ...) but has Note: -> only keep whats before
        pattern = r"(?:<i>)Notes?:"
        match = re.search(pattern, desc)
        if match:
            desc = desc[: match.start()]

        # Remove new lines
        desc = desc.replace("\n", "").replace("\r", "")

    """Genre description"""
    lang = {
        "JP": "japanese manga",
        "KR": "korean manhwa",
        "CN": "chinese manhua",
        "TW": "taiwanese manhua",
    }[row.country]
    genres = ", ".join(literal_eval(row.genres)).lower()
    genre_desc = f"A {genres} {lang} comic series."

    """Tag description"""
    tags_list = [tag[0] for tag in literal_eval(row.tags) if tag[1] / 100 >= tag_thresh]
    tags = ", ".join(tags_list).lower()
    tag_desc = f"The story contains the themes, tropes and topics of {tags}."

    full_desc = f"{desc} {genre_desc} {tag_desc}"

    return full_desc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to data csv"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="cleaned.csv",
        type=str,
        help="Output of cleaned data csv",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    raw["input"] = np.nan  # Full input descriptions
    raw.dropna(subset=["description"], inplace=True)  # Drop rows without a description
    raw.input = raw.apply(clean_description, axis=1)
    raw.to_csv(args.output, index=False)
