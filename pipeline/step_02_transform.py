import pandas as pd
from pipeline.step_00_config import CACHE_DIR
from utilities.util import parse_list

CLEAN_IN = CACHE_DIR / "cleaned.parquet"
TRANSFORMED_OUT = CACHE_DIR / "transformed.parquet"

def main():
    # Skip if transformed dataset already exists (avoids recomputation)
    if TRANSFORMED_OUT.exists():
        print("[02_transform] Cache exists. Skipping transformation.")
        return

    cleaned = pd.read_parquet(CLEAN_IN)
    transformed = user_flags(normalize(type_cast(parsing_lists(cleaned))))

    # Apply full transformation pipeline step-by-step
    transformed.to_parquet(TRANSFORMED_OUT, index=False)
    print("[02_transform] Saved:", TRANSFORMED_OUT)

def parsing_lists(merged):
    # Convert string representations of lists into actual Python lists
    merged["genres_list"] = merged["genres"].apply(parse_list)
    merged["languages_list"] = merged["spoken_languages"].apply(parse_list)

    return merged

def type_cast(merged):
    # Ensure numeric columns are properly typed; invalid values become NaN
    merged["runtime"] = pd.to_numeric(merged["runtime"], errors="coerce")
    merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce")
    merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce")
    merged["year_released"] = pd.to_numeric(merged["year_released"], errors="coerce")

    return merged


def normalize(merged):
    # Adjust ratings relative to each user's average to remove personal bias
    user_mean = merged.groupby("user")["rating"].transform("mean")
    merged["rating_centered"] = merged["rating"] - user_mean

    return merged


# Define thresholds for classifying user preference relative to their average rating
LIKE_THRESHOLD = 0.25  # significantly above user's mean
DISLIKE_THRESHOLD = -0.25 # significantly below user's mean


def user_flags(df):
    # Create binary indicators for liked/disliked items based on centered rating
    df["like_flag"] = (df["rating_centered"] > LIKE_THRESHOLD).astype(int)
    df["dislike_flag"] = (df["rating_centered"] < DISLIKE_THRESHOLD).astype(int)

    # Remove original raw columns that are no longer needed after transformation
    df.drop(columns=['genres', 'spoken_languages', 'movie_id_norm'], inplace=True)

    return df

if __name__ == "__main__":
    main()