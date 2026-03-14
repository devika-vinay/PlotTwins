import numpy as np
import pandas as pd
from pipeline.step_00_config import CACHE_DIR

TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"
USER_PROFILES_OUT = CACHE_DIR / "user_profiles.parquet"

CLASSIC_YEAR_CUTOFF = 1980
MODERN_YEAR_CUTOFF = 2010
SPECIALIST_SHARE = 0.35
INTERNATIONAL_SHARE = 0.50

# GTA-focused FSA list derived from the teammate-provided codes
GTA_FSAS = [
    # Toronto
    "M1B","M1C","M1E","M1G","M1H","M1J","M1K","M1L","M1M","M1N","M1P","M1R","M1S","M1T","M1V","M1W","M1X",
    "M2H","M2J","M2K","M2L","M2M","M2N","M2P","M2R",
    "M3A","M3B","M3C","M3H","M3J","M3K","M3L","M3M","M3N",
    "M4A","M4B","M4C","M4E","M4G","M4H","M4J","M4K","M4L","M4M","M4N","M4P","M4R","M4S","M4T","M4V","M4W","M4X","M4Y",
    "M5A","M5B","M5C","M5E","M5G","M5H","M5J","M5K","M5L","M5M","M5N","M5P","M5R","M5S","M5T","M5V","M5W","M5X",
    "M6A","M6B","M6C","M6E","M6G","M6H","M6J","M6K","M6L","M6M","M6N","M6P","M6R","M6S",
    "M7A","M7R","M7Y",
    "M8V","M8W","M8X","M8Y","M8Z",
    "M9A","M9B","M9C","M9L","M9M","M9N","M9P","M9R","M9V","M9W",

    # Durham
    "L1B","L1C","L1E","L1G","L1H","L1J","L1K","L1L","L1M","L1N","L1P","L1R","L1S","L1T","L1V","L1W","L1X","L1Y","L1Z",

    # York / North GTA
    "L3L","L3P","L3R","L3S","L3T","L3X","L3Y",
    "L4A","L4B","L4C","L4E","L4G","L4H","L4J","L4K","L4L","L4S",

    # Peel
    "L4T","L4V","L4W","L4X","L4Y","L4Z",
    "L5A","L5B","L5C","L5E","L5G","L5H","L5J","L5K","L5L","L5M","L5N","L5P","L5R","L5S","L5T","L5V","L5W",
    "L6P","L6R","L6S","L6T","L6V","L6W","L6X","L6Y","L6Z","L7A",

    # Vaughan / Maple / Markham area
    "L6A","L6B","L6C","L6E","L6G",

    # Halton
    "L6H","L6J","L6K","L6L","L6M",
    "L7B","L7C","L7E","L7G","L7J","L7K","L7L","L7M","L7N","L7P","L7R","L7S","L7T",
    "L8B","L9E","L9T"
]

def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy over a distribution derived from counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _top_value_and_share(series: pd.Series):
    """Return (top_value, share_of_total)."""
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return (None, 0.0)
    top_val = vc.index[0]
    share = float(vc.iloc[0] / vc.sum())
    return (top_val, share)

def _fav_decade(years: pd.Series):
    years = years.dropna().astype(int)
    if years.empty:
        return np.nan
    decades = (years // 10) * 10
    return int(decades.value_counts().idxmax())

def assign_region(fsa: str) -> str:
    if fsa.startswith("M"):
        return "Toronto"
    if fsa.startswith("L1"):
        return "Durham"
    if fsa.startswith(("L3", "L4", "L6A", "L6B", "L6C", "L6E", "L6G")):
        return "York"
    if fsa.startswith(("L4T", "L4V", "L4W", "L4X", "L4Y", "L4Z", "L5", "L6P", "L6R", "L6S", "L6T", "L6V", "L6W", "L6X", "L6Y", "L6Z", "L7A")):
        return "Peel"
    if fsa.startswith(("L6H", "L6J", "L6K", "L6L", "L6M", "L7", "L8B", "L9E", "L9T")):
        return "Halton"
    return "GTA"

def main():
    if USER_PROFILES_OUT.exists():
        print("[03_user_profiles] Cache exists. Skipping user profile build.")
        return

    df = pd.read_parquet(TRANSFORMED_IN)
    rng = np.random.default_rng(42)

    # ---- basic per-user rating behavior ----
    base = df.groupby("user").agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        mean_centered=("rating_centered", "mean"),
        std_centered=("rating_centered", "std"),
        like_rate=("like_flag", "mean"),
        avg_release_year=("year_released", "mean"),
        avg_vote_count=("vote_count", "mean"),
        avg_vote_average=("vote_average", "mean"),
    ).reset_index()

    # ---- time preference flags ----
    # classic_share / modern_share computed from interaction rows (not grouped agg directly)
    year = df[["user", "year_released"]].copy()
    year["is_classic"] = (year["year_released"] < CLASSIC_YEAR_CUTOFF).astype(int)
    year["is_modern"] = (year["year_released"] >= MODERN_YEAR_CUTOFF).astype(int)

    year_stats = year.groupby("user").agg(
        classic_share=("is_classic", "mean"),
        modern_share=("is_modern", "mean"),
        fav_decade=("year_released", _fav_decade),
    ).reset_index()

    # ---- genre features ----
    g = df[["user", "genres_list"]].explode("genres_list").dropna(subset=["genres_list"])
    g_counts = g.groupby(["user", "genres_list"]).size().rename("cnt").reset_index()

    # top_genre + share
    top_genre = g_counts.groupby("user").apply(
        lambda x: pd.Series({
            "top_genre": x.sort_values("cnt", ascending=False).iloc[0]["genres_list"],
            "top_genre_share": float(x["cnt"].max() / x["cnt"].sum()),
            "genre_entropy": _entropy_from_counts(x["cnt"].to_numpy()),
            "n_unique_genres": int(x["genres_list"].nunique()),
        })
    ).reset_index()

    # ---- language features ----
    l = df[["user", "languages_list"]].explode("languages_list").dropna(subset=["languages_list"])
    l_counts = l.groupby(["user", "languages_list"]).size().rename("cnt").reset_index()

    top_lang = l_counts.groupby("user").apply(
        lambda x: pd.Series({
            "top_language": x.sort_values("cnt", ascending=False).iloc[0]["languages_list"],
            "top_language_share": float(x["cnt"].max() / x["cnt"].sum()),
            "n_unique_languages": int(x["languages_list"].nunique()),
        })
    ).reset_index()

    # english share (robust to different spellings — keep simple)
    l["is_english"] = l["languages_list"].astype(str).str.lower().isin(["english", "en", "eng"]).astype(int)
    english_share = l.groupby("user")["is_english"].mean().rename("english_share").reset_index()

    # ---- merge all ----
    user_profiles = base.merge(year_stats, on="user", how="left")
    user_profiles = user_profiles.merge(top_genre, on="user", how="left")
    user_profiles = user_profiles.merge(top_lang, on="user", how="left")
    user_profiles = user_profiles.merge(english_share, on="user", how="left")

    # fill for users with missing lists
    for col in ["classic_share", "modern_share", "english_share", "top_genre_share", "top_language_share",
                "genre_entropy", "n_unique_genres", "n_unique_languages"]:
        if col in user_profiles.columns:
            user_profiles[col] = user_profiles[col].fillna(0)

    # ---- persona tags (simple rules) ----
    def tags(row):
        t = []

        # Classic vs modern
        if row.get("classic_share", 0) >= 0.40:
            t.append("old-timey classic lover")
        if row.get("modern_share", 0) >= 0.60:
            t.append("modern release chaser")

        # Genre specialist vs omnivore
        if row.get("top_genre_share", 0) >= SPECIALIST_SHARE:
            t.append("genre specialist")
        if row.get("n_unique_genres", 0) >= 8 and row.get("top_genre_share", 0) < 0.25:
            t.append("genre omnivore")

        # International watcher
        non_english_share = 1 - row.get("english_share", 0)
        if non_english_share >= INTERNATIONAL_SHARE:
            t.append("international watcher")

        # Rating style
        if row.get("mean_rating", 0) >= 4.0:
            t.append("high rater")
        if row.get("mean_rating", 5) <= 3.0:
            t.append("tough critic")

        # Niche-ish proxy
        if row.get("avg_vote_count", 999999) <= 200:
            t.append("niche explorer")

        return t

    user_profiles["persona_tags"] = user_profiles.apply(tags, axis=1)

    # ---- assign synthetic GTA location ----
    user_profiles["fsa"] = rng.choice(GTA_FSAS, size=len(user_profiles), replace=True)
    user_profiles["region"] = user_profiles["fsa"].apply(assign_region)

    # cache
    user_profiles.to_parquet(USER_PROFILES_OUT, index=False)
    print("[03_user_profiles] Saved:", USER_PROFILES_OUT)
    print(user_profiles.head())

if __name__ == "__main__":
    main()


