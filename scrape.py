import math
import time

import pandas as pd
import requests
from tqdm import tqdm


def grab_page(page: int, n: int = 50):
    query = """
    query ($page: Int, $perPage: Int) {
        Page (page: $page, perPage: $perPage) {
            pageInfo {
                total
                currentPage
                hasNextPage
            }
            media(sort: POPULARITY_DESC, type: MANGA) {
                title {
                    romaji
                    english
                    native
                }
                description
                tags {
                    name
                    rank
                }
                siteUrl
                countryOfOrigin
                    genres
                    coverImage {
                        extraLarge
                    }
                }
        }
    }
    """
    variables = {"page": page, "perPage": n}
    url = "https://graphql.anilist.co"

    response = requests.post(url, json={"query": query, "variables": variables}).json()
    data = response["data"]["Page"]["media"]
    return data


def construct_rows(data):
    rows = []
    for d in data:
        tags = [(t["name"], t["rank"]) for t in d["tags"]]
        rows.append(
            {
                "romaji": d["title"]["romaji"],
                "english": d["title"]["english"],
                "native": d["title"]["native"],
                "description": d["description"],
                "tags": tags,
                "url": d["siteUrl"],
                "country": d["countryOfOrigin"],
                "genres": d["genres"],
                "cover": d["coverImage"]["extraLarge"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    total = 1000
    n = 50
    wait = 0
    out_path = "raw.csv"

    df = pd.DataFrame(
        columns=[
            "romaji",
            "english",
            "native",
            "description",
            "tags",
            "url",
            "country",
            "genres",
            "cover",
        ]
    )
    for i in tqdm(range(math.ceil(total / n))):
        page_data = grab_page(i + 1, n)
        rows = construct_rows(page_data)
        df = pd.concat([df, rows], ignore_index=True)
        time.sleep(wait)  # Rate limit is 90 requests (rows) per minute

    df.to_csv(out_path)
