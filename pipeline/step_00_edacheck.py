import pandas as pd
from pipeline.step_00_config import CACHE_DIR

# Path to the transformed dataset (assumed to be preprocessed earlier in the pipeline)
TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"

def main():
    # Load the transformed dataset from a parquet file into a DataFrame
    merged = pd.read_parquet(TRANSFORMED_IN)

    # Print the number of unique users in the dataset
    print("Unique users:", merged["user"].nunique())

    # Expand the 'genres_list' column so each genre gets its own row
    # Then drop rows where genre is missing (NaN)
    exploded = merged.explode("genres_list").dropna(subset=["genres_list"])

    # Count how many ratings each user has given
    user_movie_counts = merged.groupby("user").size()
    print("Ratings per user (describe):")
    # Show summary statistics (mean, min, max, etc.) of ratings per user
    print(user_movie_counts.describe())

    # Compute per-user rating statistics: average, min, max, and total count
    user_stats = merged.groupby("user")["rating"].agg(["mean","min","max","count"])

    # Identify users who mostly give high ratings (average rating >= 4)
    always_high = user_stats[user_stats["mean"] >= 4]

    # Identify users who never give high ratings (max rating <= 3)
    never_high = user_stats[user_stats["max"] <= 3]

    # Print counts of these user groups
    print("Users rating mostly 4-5:", len(always_high))
    print("Users never rating above 3:", len(never_high))

    # Count how often each genre appears
    genre_counts = exploded["genres_list"].value_counts()

    # Convert counts into percentages of total genre occurrences
    genre_percent = genre_counts / genre_counts.sum() * 100
    
    # Identify rare genres (less than 1% of all genre occurrences)
    rare_genres = genre_percent[genre_percent < 1]
    print("Rare genres (<1%):")
    print(rare_genres)

# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()