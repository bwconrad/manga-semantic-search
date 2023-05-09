import math
from argparse import ArgumentParser
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
    parser = ArgumentParser()
    parser.add_argument("-n", default=1000, type=int, help="Scrape the top-n series")
    parser.add_argument(
        "--batch", "-b", default=50, type=int, help="Batch size per API request"
    )
    parser.add_argument(
        "--wait",
        "-w",
        required=True,
        type=int,
        help="Number of seconds to wait between requests. Use to deal with rate limit.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="raw.csv",
        type=str,
        help="Output path of scraped data",
    )
    args = parser.parse_args()

    total = args.n
    batch = args.batch
    wait = args.wait
    out_path = args.output

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
    for i in tqdm(range(math.ceil(total / batch))):
        page_data = grab_page(i + 1, batch)
        rows = construct_rows(page_data)
        df = pd.concat([df, rows], ignore_index=True)
        time.sleep(wait)  # Rate limit is 90 requests (rows) per minute

    df.to_csv(out_path)
    print(f"Finished scraping. Results written to {out_path}")
