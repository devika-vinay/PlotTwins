import pandas as pd
from datetime import datetime

from pipeline.step_00_config import CACHE_DIR

# ── Input/Output Paths ──────────────────────────────────────────────
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet" # user → cluster assignments
CLUSTER_PERSONAS_IN = CACHE_DIR / "cluster_personas.parquet" # user → cluster assignments
CLUSTER_MOVIE_KB_IN = CACHE_DIR / "cluster_movie_kb.parquet" # cluster-level movie knowledge base

EVENT_SUGGESTIONS_OUT = CACHE_DIR / "event_suggestions.parquet" # cluster-level movie knowledge base

# ── Season & Date Utilities ─────────────────────────────────────────
def get_current_season(dt: datetime) -> str:
    month = dt.month
    """Return the current season (winter/spring/summer/fall) based on month"""
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "fall"
# ── FSA Cluster Demand ───────────────────────────────────────────────
def build_fsa_cluster_demand(cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cluster distribution within each FSA (Forward Sortation Area)
    Returns per-FSA per-cluster counts and share of total FSA users
    """
    # Count users per FSA + cluster
    fsa_cluster_counts = (
        cluster_assignments
        .groupby(["fsa", "cluster"], as_index=False)
        .agg(n_users=("user", "count"))
    )
    # Total users per FSA
    fsa_totals = (
        cluster_assignments
        .groupby("fsa", as_index=False)
        .agg(total_users_in_fsa=("user", "count"))
    )
    # Merge counts with totals
    fsa_cluster_demand = fsa_cluster_counts.merge(
        fsa_totals,
        on="fsa",
        how="left"
    )

    fsa_cluster_demand["cluster_share_in_fsa"] = (
        fsa_cluster_demand["n_users"] / fsa_cluster_demand["total_users_in_fsa"]
    )

    fsa_cluster_demand = fsa_cluster_demand.sort_values(
        by=["fsa", "n_users"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return fsa_cluster_demand

def attach_cluster_personas(
    fsa_cluster_demand: pd.DataFrame,
    cluster_personas: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach persona metadata (name, top genres) to each FSA-cluster demand row
    """
    persona_cols = [
        "cluster",
        "persona_name",
        "top_genres",
    ]

    enriched = fsa_cluster_demand.merge(
        cluster_personas[persona_cols],
        on="cluster",
        how="left"
    )

    return enriched
# ── Top Movies by Cluster ───────────────────────────────────────────
def build_top_movies_by_cluster(
        
    cluster_movie_kb: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    For each cluster, return top N representative movies
    based on representative_score (descending)
    """
    movie_cols = [
        "cluster",
        "display_title",
        "year_released",
        "popularity_bucket",
        "representative_score",
    ]

    top_movies = (
        cluster_movie_kb[movie_cols]
        .sort_values(
            by=["cluster", "representative_score"],
            ascending=[True, False]
        )
        .groupby("cluster", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return top_movies
# ── Seasonal Event Themes ───────────────────────────────────────────
def get_seasonal_event_themes(dt: datetime) -> list[dict]:
    """
    Return a list of event themes for the current date.
    Month-specific overrides (e.g., Halloween, December holidays) take precedence.
    Otherwise, default to seasonal themes.
    """
    month = dt.month

    # Month-specific overrides first
    if month == 10:
        return [
            {"theme_key": "halloween_horror", "theme_label": "Halloween Horror Night"},
            {"theme_key": "gothic_classics", "theme_label": "Gothic Classics Screening"},
            {"theme_key": "psychological_thrillers", "theme_label": "Psychological Thriller Weekend"},
        ]

    if month == 12:
        return [
            {"theme_key": "holiday_family", "theme_label": "Holiday Family Classics"},
            {"theme_key": "winter_fantasy", "theme_label": "Winter Fantasy Adventure"},
            {"theme_key": "cozy_romance", "theme_label": "Cozy Holiday Romance"},
        ]

    # Season-level defaults
    season = get_current_season(dt)

    if season == "spring":
        return [
            {"theme_key": "adventure_revival", "theme_label": "Spring Adventure Revival"},
            {"theme_key": "feel_good_family", "theme_label": "Feel-Good Family Favorites"},
            {"theme_key": "sci_fi_refresh", "theme_label": "Sci-Fi Spotlight"},
        ]

    if season == "summer":
        return [
            {"theme_key": "summer_blockbusters", "theme_label": "Summer Blockbuster Throwback"},
            {"theme_key": "action_adventure", "theme_label": "Action Adventure Night"},
            {"theme_key": "fantasy_escape", "theme_label": "Fantasy Escape Weekend"},
        ]

    if season == "fall":
        return [
            {"theme_key": "dark_cinema", "theme_label": "Dark Cinema Weekend"},
            {"theme_key": "mystery_thriller", "theme_label": "Mystery and Thriller Spotlight"},
            {"theme_key": "prestige_drama", "theme_label": "Prestige Drama Revival"},
        ]

    # winter default
    return [
        {"theme_key": "winter_classics", "theme_label": "Winter Classics Series"},
        {"theme_key": "family_matinees", "theme_label": "Family Matinee Picks"},
        {"theme_key": "fantasy_wonder", "theme_label": "Fantasy Wonder Weekend"},
    ]


def theme_matches_cluster(theme_key: str, top_genres_text: str) -> bool:
    """
    Determine if an event theme matches a cluster's top genres
    """
    genres_text = str(top_genres_text).lower()

    theme_keywords = {
        "halloween_horror": ["horror", "thriller", "mystery"],
        "gothic_classics": ["horror", "drama", "mystery", "fantasy"],
        "psychological_thrillers": ["thriller", "mystery", "crime"],
        "holiday_family": ["family", "animation", "comedy", "fantasy"],
        "winter_fantasy": ["fantasy", "adventure", "family"],
        "cozy_romance": ["romance", "comedy", "drama"],
        "adventure_revival": ["adventure", "action", "fantasy"],
        "feel_good_family": ["family", "animation", "comedy"],
        "sci_fi_refresh": ["science fiction", "fantasy", "adventure"],
        "summer_blockbusters": ["action", "adventure", "science fiction"],
        "action_adventure": ["action", "adventure", "fantasy"],
        "fantasy_escape": ["fantasy", "adventure", "science fiction"],
        "dark_cinema": ["thriller", "horror", "crime", "mystery"],
        "mystery_thriller": ["mystery", "thriller", "crime"],
        "prestige_drama": ["drama", "war", "crime"],
        "winter_classics": ["drama", "family", "fantasy"],
        "family_matinees": ["family", "animation", "comedy"],
        "fantasy_wonder": ["fantasy", "adventure", "family"],
    }

    keywords = theme_keywords.get(theme_key, [])
    return any(keyword in genres_text for keyword in keywords)
# ── Event Suggestion Construction ─────────────────────────────────────
def build_event_suggestion_rows(
    enriched_demand: pd.DataFrame,
    top_movies_by_cluster: pd.DataFrame,
    top_n_movies: int = 3
) -> pd.DataFrame:
    """
    For each FSA + cluster combination with matching seasonal themes,
    assign top N representative movies and expand rows by theme
    """
    eligible = enriched_demand[
        enriched_demand["season_theme_matches"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()

    top_movies_trimmed = (
        top_movies_by_cluster
        .sort_values(by=["cluster", "representative_score"], ascending=[True, False])
        .groupby("cluster", as_index=False)
        .head(top_n_movies)
        .reset_index(drop=True)
    )

    merged = eligible.merge(
        top_movies_trimmed,
        on="cluster",
        how="left"
    )

    merged = merged.explode("season_theme_matches").rename(
        columns={"season_theme_matches": "event_theme"}
    )

    merged = merged[
        [
            "fsa",
            "cluster",
            "cluster_share_in_fsa",
            "persona_name",
            "top_genres",
            "event_theme",
            "display_title",
            "year_released",
            "popularity_bucket",
            "representative_score",
        ]
    ].reset_index(drop=True)

    return merged
# ── Ranking and Deduplication ───────────────────────────────────────
def rank_event_suggestions(
    event_suggestion_rows: pd.DataFrame,
    top_k_per_fsa: int = 5
) -> pd.DataFrame:
    """
    Score suggestions using cluster share, representative score, and popularity bonus.
    Keep only top K per FSA.
    """
    ranked = event_suggestion_rows.copy()

    popularity_bonus_map = {
        "high": 0.10,
        "mid": 0.05,
        "low": 0.00,
    }

    ranked["cluster_share_in_fsa"] = pd.to_numeric(
        ranked["cluster_share_in_fsa"],
        errors="coerce"
    )

    ranked["representative_score"] = pd.to_numeric(
        ranked["representative_score"],
        errors="coerce"
    )

    ranked["popularity_bonus"] = (
        ranked["popularity_bucket"]
        .astype(str)
        .map(popularity_bonus_map)
        .fillna(0.0)
    )

    ranked["business_score"] = (
        ranked["cluster_share_in_fsa"] * 0.60
        + ranked["representative_score"] * 0.03
        + ranked["popularity_bonus"]
    )

    ranked = ranked.sort_values(
        by=["fsa", "business_score", "cluster_share_in_fsa", "representative_score"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    ranked["rank_within_fsa"] = ranked.groupby("fsa").cumcount() + 1

    ranked = ranked[ranked["rank_within_fsa"] <= top_k_per_fsa].copy()

    return ranked

def deduplicate_event_suggestions(
    ranked_event_suggestions: pd.DataFrame,
    top_k_per_fsa: int = 5
) -> pd.DataFrame:
    """
    Remove duplicate movie titles per FSA, keeping the highest-scoring instance
    """
    deduped = ranked_event_suggestions.copy()

    deduped = deduped.sort_values(
        by=["fsa", "business_score", "cluster_share_in_fsa", "representative_score"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    deduped = deduped.drop_duplicates(
        subset=["fsa", "display_title"],
        keep="first"
    ).copy()

    deduped["final_rank_within_fsa"] = deduped.groupby("fsa").cumcount() + 1

    deduped = deduped[deduped["final_rank_within_fsa"] <= top_k_per_fsa].copy()

    return deduped
# ── Business Explanation ───────────────────────────────────────────
def build_business_explanation(row) -> str:
    """
    Human-readable explanation of why a movie/event was suggested for this FSA
    """
    cluster_share_pct = round(float(row["cluster_share_in_fsa"]) * 100, 1)

    return (
        f"Suggested for FSA {row['fsa']} because {row['persona_name']} "
        f"represents {cluster_share_pct}% of local users in this area, "
        f"the event theme '{row['event_theme']}' matches this cluster's taste profile, "
        f"and '{row['display_title']}' is a strong representative movie for that audience."
    )
# ── MAIN ────────────────────────────────────────────────────────────
def main():
    # Skip if cached
    if EVENT_SUGGESTIONS_OUT.exists():
        print("[13_event_suggestions] Cache exists. Skipping event suggestion build.")
        return

    if not CLUSTER_ASSIGNMENTS_IN.exists():
        raise FileNotFoundError(
            "[13_event_suggestions] Missing input: cluster_assignments.parquet"
        )

    if not CLUSTER_PERSONAS_IN.exists():
        raise FileNotFoundError(
            "[13_event_suggestions] Missing input: cluster_personas.parquet"
        )

    if not CLUSTER_MOVIE_KB_IN.exists():
        raise FileNotFoundError(
            "[13_event_suggestions] Missing input: cluster_movie_kb.parquet"
        )
    # Load inputs
    cluster_assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)
    cluster_personas = pd.read_parquet(CLUSTER_PERSONAS_IN)
    cluster_movie_kb = pd.read_parquet(CLUSTER_MOVIE_KB_IN)

    today = datetime.today()

    print("[13_event_suggestions] Loaded cluster_assignments:", cluster_assignments.shape)
    print("[13_event_suggestions] Loaded cluster_personas:", cluster_personas.shape)
    print("[13_event_suggestions] Loaded cluster_movie_kb:", cluster_movie_kb.shape)
    # Build FSA-level demand
    fsa_cluster_demand = build_fsa_cluster_demand(cluster_assignments)
    # Match seasonal themes to cluster genres
    enriched_demand = attach_cluster_personas(
        fsa_cluster_demand,
        cluster_personas
    )

    top_movies_by_cluster = build_top_movies_by_cluster(
        cluster_movie_kb,
        top_n=10
    )

    seasonal_themes = get_seasonal_event_themes(today)

    enriched_demand["season_theme_matches"] = enriched_demand.apply(
        lambda row: [
            theme["theme_label"]
            for theme in seasonal_themes
            if theme_matches_cluster(theme["theme_key"], row["top_genres"])
        ],
        axis=1
    )
    # Construct event suggestions
    event_suggestion_rows = build_event_suggestion_rows(
        enriched_demand=enriched_demand,
        top_movies_by_cluster=top_movies_by_cluster,
        top_n_movies=3
    )
    # Rank and deduplicate
    ranked_event_suggestions = rank_event_suggestions(
        event_suggestion_rows,
        top_k_per_fsa=5
    )

    final_event_suggestions = deduplicate_event_suggestions(
        ranked_event_suggestions,
        top_k_per_fsa=5
    )
    # Add business explanation
    final_event_suggestions["business_explanation"] = final_event_suggestions.apply(
        build_business_explanation,
        axis=1
    )

    final_event_suggestions.to_parquet(EVENT_SUGGESTIONS_OUT, index=False)
    print(f"[13_event_suggestions] Saved event suggestions: {final_event_suggestions.shape}")

if __name__ == "__main__":
    main()