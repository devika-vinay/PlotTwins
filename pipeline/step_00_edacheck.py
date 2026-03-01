import pandas as pd
from pipeline.step_00_config import CACHE_DIR

TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"

def main():
    merged = pd.read_parquet(TRANSFORMED_IN)

    print("Unique users:", merged["user"].nunique())

    exploded = merged.explode("genres_list").dropna(subset=["genres_list"])

    user_movie_counts = merged.groupby("user").size()
    print("Ratings per user (describe):")
    print(user_movie_counts.describe())

    user_stats = merged.groupby("user")["rating"].agg(["mean","min","max","count"])
    always_high = user_stats[user_stats["mean"] >= 4]
    never_high = user_stats[user_stats["max"] <= 3]
    print("Users rating mostly 4-5:", len(always_high))
    print("Users never rating above 3:", len(never_high))

    genre_counts = exploded["genres_list"].value_counts()
    genre_percent = genre_counts / genre_counts.sum() * 100
    rare_genres = genre_percent[genre_percent < 1]
    print("Rare genres (<1%):")
    print(rare_genres)

if __name__ == "__main__":
    main()