import numpy as np
import pandas as pd
from pipeline.step_00_config import CACHE_DIR

TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"
USER_PROFILES_OUT = CACHE_DIR / "user_profiles.parquet"

CLASSIC_YEAR_CUTOFF = 1980
MODERN_YEAR_CUTOFF = 2010
SPECIALIST_SHARE = 0.35
INTERNATIONAL_SHARE = 0.50

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

def main():
    if USER_PROFILES_OUT.exists():
        print("[03_user_profiles] Cache exists. Skipping user profile build.")
        return

    df = pd.read_parquet(TRANSFORMED_IN)

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

    # cache
    user_profiles.to_parquet(USER_PROFILES_OUT, index=False)
    print("[03_user_profiles] Saved:", USER_PROFILES_OUT)
    print(user_profiles.head())

if __name__ == "__main__":
    main()