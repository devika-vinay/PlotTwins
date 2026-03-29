import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline.step_00_config import CACHE_DIR
from utilities.util import parse_list

CLEAN_IN = CACHE_DIR / "cleaned.parquet"
TRANSFORMED_OUT = CACHE_DIR / "transformed.parquet"


def main():
    if TRANSFORMED_OUT.exists():
        print("[02_transform] Cache exists. Skipping transformation.")
        return

    cleaned = pd.read_parquet(CLEAN_IN)
    transformed = user_flags(normalize(type_cast(parsing_lists(cleaned))))

    transformed.to_parquet(TRANSFORMED_OUT, index=False)
    print("[02_transform] Saved:", TRANSFORMED_OUT)


def parsing_lists(merged):
    # Convert stringified lists to actual lists
    merged["genres_list"] = merged["genres"].apply(parse_list)
    merged["languages_list"] = merged["spoken_languages"].apply(parse_list)

    return merged


def type_cast(merged):
    # Convert numeric columns to appropriate types, coercing errors to NaN
    merged["runtime"] = pd.to_numeric(merged["runtime"], errors="coerce")
    merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce")
    merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce")
    merged["year_released"] = pd.to_numeric(merged["year_released"], errors="coerce")

    return merged


def normalize(merged):
    # Center ratings by user mean to account for individual rating biases
    user_mean = merged.groupby("user")["rating"].transform("mean")
    merged["rating_centered"] = merged["rating"] - user_mean

    return merged


# Use a tolerance band — neutral zone around the mean
LIKE_THRESHOLD = 0.25  # at least 0.25 stars above personal mean
DISLIKE_THRESHOLD = -0.25


def user_flags(df):
    # Create binary flags for like/dislike based on centered ratings
    df["like_flag"] = (df["rating_centered"] > LIKE_THRESHOLD).astype(int)
    df["dislike_flag"] = (df["rating_centered"] < DISLIKE_THRESHOLD).astype(int)

    # Drop original list columns and movie_id_norm as they are no longer needed
    df.drop(columns=['genres', 'spoken_languages', 'movie_id_norm'], inplace=True)

    return df


if __name__ == "__main__":
    main()
