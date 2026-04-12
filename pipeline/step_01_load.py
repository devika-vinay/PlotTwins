import numpy as np
import pandas as pd
from pipeline.step_00_config import CACHE_DIR, MOVIES_CSV_PATH, RATINGS_PARQUET_PATH

REDUCED_OUT = CACHE_DIR / "cleaned.parquet"


def main():
    # Skip processing if cleaned dataset already exists (caching for efficiency)
    if REDUCED_OUT.exists():
        print("[01_load] Cache exists. Skipping load, reduce and clean.")
        return
    
    merged_df = merge()
    cleaned_df = clean(drop_unusable(merged_df))

    # Save cleaned dataset to cache
    cleaned_df.to_parquet(REDUCED_OUT, index=False)
    print("[01_load] Saved:", REDUCED_OUT)
    

def merge():
    # Load datasets
    movies = pd.read_csv(
        MOVIES_CSV_PATH,
        engine="python",
        on_bad_lines="skip" # Skip malformed rows in CSV
    )
    movies_2 = pd.read_parquet(RATINGS_PARQUET_PATH)

    # Normalize keys for safer merging (avoid case/whitespace mismatches)
    movies["movie_id_norm"] = movies["movie_id"].astype(str).str.strip().str.lower()
    movies_2["title_norm"] = movies_2["title"].astype(str).str.strip().str.lower()

    # Merge ratings with movie metadata
    merged = movies_2.merge(
        movies,
        left_on="title_norm",
        right_on="movie_id_norm",
        how="left"
    )

    # Merge results analysis
    # print("rows in ratings:", len(movies_2))
    # print("rows after merge:", len(merged))
    # print("matched movie_title:", merged["movie_title"].notna().sum())
    # print("unmatched:", merged["movie_title"].isna().sum())

    return merged


def drop_unusable(merged_df):
    # Remove irrelevant or redundant columns after merge
    drop_cols = ['movie_id_x', 'title', '_id', 'image_url', 'imdb_id', 'imdb_link', 'movie_id_y', 'movie_title',
                'production_countries', 'release_date', 'tmdb_id', 'tmdb_link']

    merged = merged_df.drop(columns=drop_cols)

    # Missing value analysis
    # na_counts = merged.isna().sum().sort_values(ascending=False)
    # na_percent = (merged.isna().mean() * 100).sort_values(ascending=False)

    # missing_table = pd.concat([na_counts, na_percent], axis=1)
    # missing_table.columns = ["missing_count", "missing_percent"]

    # Drop rows with any missing values to ensure clean downstream analysis
    merged = merged.dropna()
    # print("Rows after dropping ALL NaNs:", len(merged))

    return merged

def clean(df):
    df["runtime"] = df["runtime"].replace(0, np.nan)
    df["vote_average"] = df["vote_average"].apply(lambda x: x if 0 <= x <= 5 else np.nan)

    valid_ratings = np.arange(0.5, 5.1, 0.5)
    df = df[df["rating"].isin(valid_ratings)]

    return df

if __name__ == "__main__":
    main()