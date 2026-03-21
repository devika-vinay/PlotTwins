import itertools
import math
import pandas as pd
import numpy as np

from pipeline.step_00_config import CACHE_DIR


FEATURE_MATRIX_IN = CACHE_DIR / "feature_matrix.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"
USER_MATCHES_IN = CACHE_DIR / "user_matches.parquet"

CLUSTER_PERSONAS_OUT = CACHE_DIR / "cluster_personas.parquet"
USER_DASHBOARD_OUT = CACHE_DIR / "user_dashboard.parquet"


PERSONA_LIBRARY = [
    {
        "persona_key": "modern_animation_family",
        "persona_name": "Modern Animation & Family Entertainment",
        "short_label": "Modern Animation & Family Fans",
        "expected_genres": ["Animation", "Family", "Science Fiction", "Comedy", "Adventure"],
        "expected_eras": ["era_2010.0", "era_2020.0"],
        "expected_pop": "pop_low",
        "interpretation": (
            "You enjoy modern animated and family-friendly films, often with imaginative "
            "or sci-fi elements. You tend to explore newer releases, sometimes outside blockbuster hits."
        ),
        "example_movies": ["Spider-Man: Into the Spider-Verse", "Coco", "The Lego Movie"],
    },
    {
        "persona_key": "classic_fantasy_adventure",
        "persona_name": "Classic Fantasy & Adventure Fans",
        "short_label": "Classic Fantasy / Adventure Lovers",
        "expected_genres": ["Comedy", "Fantasy", "Adventure", "Family", "Animation"],
        "expected_eras": ["era_1970.0", "era_1980.0", "era_1990.0"],
        "expected_pop": "pop_low",
        "interpretation": (
            "You enjoy classic fantasy adventures and nostalgic films, often from earlier decades."
        ),
        "example_movies": ["The Princess Bride", "Back to the Future", "The NeverEnding Story"],
    },
    {
        "persona_key": "suspense_mystery",
        "persona_name": "Suspense & Mystery Viewers",
        "short_label": "Suspense & Mystery Viewers",
        "expected_genres": ["Thriller", "Mystery", "Horror", "TV Movie"],
        "expected_eras": ["era_1970.0", "era_1980.0", "era_1990.0"],
        "expected_pop": "pop_mid",
        "interpretation": (
            "You gravitate toward suspenseful storytelling, including thrillers and mysteries, "
            "often from the late 20th century."
        ),
        "example_movies": ["Se7en", "The Sixth Sense", "Twin Peaks"],
    },
    {
        "persona_key": "classic_cinema",
        "persona_name": "Classic Cinema Enthusiasts",
        "short_label": "Classic Cinema Enthusiasts",
        "expected_genres": ["Thriller", "Horror", "Mystery", "Romance"],
        "expected_eras": ["era_1940.0", "era_1950.0", "era_1960.0"],
        "expected_pop": "pop_low",
        "interpretation": (
            "You appreciate vintage cinema, including noir, early thrillers, and classic dramas."
        ),
        "example_movies": ["Casablanca", "Psycho", "Rear Window"],
    },
    {
        "persona_key": "blockbuster_adventure",
        "persona_name": "Blockbuster Adventure Fans",
        "short_label": "Blockbuster Adventure Fans",
        "expected_genres": ["Fantasy", "Adventure", "Action", "Family", "Science Fiction"],
        "expected_eras": ["era_2000.0", "era_2010.0"],
        "expected_pop": "pop_high",
        "interpretation": (
            "You enjoy large-scale action and fantasy franchises, typically modern blockbuster entertainment."
        ),
        "example_movies": ["Avengers", "Harry Potter", "Lord of the Rings"],
    },
    {
        "persona_key": "serious_drama_crime",
        "persona_name": "Serious Drama & Crime Fans",
        "short_label": "Serious Drama & Crime Fans",
        "expected_genres": ["Drama", "Crime", "Thriller", "Mystery", "War"],
        "expected_eras": ["era_2000.0"],
        "expected_pop": "pop_high",
        "interpretation": (
            "You prefer serious, narrative-driven films, including crime dramas and war stories."
        ),
        "example_movies": ["The Dark Knight", "The Departed", "Saving Private Ryan"],
    },
]


def safe_top_from_prefix(row: pd.Series, prefix: str, top_n: int = 3):
    cols = [c for c in row.index if c.startswith(prefix)]
    if not cols:
        return []
    values = row[cols].sort_values(ascending=False).head(top_n)
    cleaned = []
    for col in values.index:
        cleaned.append(col.replace(prefix, ""))
    return cleaned


def describe_popularity(row: pd.Series):
    pop_cols = [c for c in ["pop_low", "pop_mid", "pop_high"] if c in row.index]
    if not pop_cols:
        return "unknown"
    top = row[pop_cols].sort_values(ascending=False).index[0]
    return top


def describe_behavior(row: pd.Series):
    like_rate = row.get("like_rate", np.nan)
    dislike_rate = row.get("dislike_rate", np.nan)

    if pd.notna(like_rate) and pd.notna(dislike_rate):
        if like_rate >= 0.65 and dislike_rate <= 0.20:
            return "strongly positive"
        if like_rate >= 0.50 and dislike_rate <= 0.30:
            return "slightly positive"
        if dislike_rate >= 0.45:
            return "more critical"
        return "neutral"
    if pd.notna(like_rate):
        if like_rate >= 0.65:
            return "strongly positive"
        if like_rate >= 0.50:
            return "slightly positive"
        if like_rate <= 0.30:
            return "more critical"
        return "neutral"
    return "unknown"


def summarize_cluster(row: pd.Series):
    return {
        "top_genres": safe_top_from_prefix(row, "genre_mean_", top_n=5),
        "top_eras": safe_top_from_prefix(row, "era_", top_n=3),
        "popularity_pref": describe_popularity(row),
        "behavior_profile": describe_behavior(row),
    }


def score_persona(summary: dict, persona: dict) -> float:
    score = 0.0

    top_genres = set(summary["top_genres"])
    expected_genres = set(persona["expected_genres"])
    genre_overlap = len(top_genres.intersection(expected_genres))
    score += genre_overlap * 3.0

    top_eras = set([f"era_{x}" if not str(x).startswith("era_") else str(x) for x in summary["top_eras"]])
    expected_eras = set(persona["expected_eras"])
    era_overlap = len(top_eras.intersection(expected_eras))
    score += era_overlap * 2.5

    if summary["popularity_pref"] == persona["expected_pop"]:
        score += 2.0

    # light behavior weighting only
    behavior = summary["behavior_profile"]
    if persona["persona_key"] == "classic_cinema" and behavior in ["more critical", "neutral"]:
        score += 1.0
    if persona["persona_key"] == "suspense_mystery" and behavior in ["slightly positive", "neutral"]:
        score += 1.0
    if persona["persona_key"] == "modern_animation_family" and behavior in ["neutral", "slightly positive"]:
        score += 1.0
    if persona["persona_key"] == "blockbuster_adventure" and behavior in ["neutral", "slightly positive", "strongly positive"]:
        score += 1.0
    if persona["persona_key"] == "serious_drama_crime" and behavior in ["neutral", "slightly positive"]:
        score += 1.0
    if persona["persona_key"] == "classic_fantasy_adventure" and behavior in ["neutral", "more critical", "slightly positive"]:
        score += 1.0

    return score


def best_unique_persona_assignment(cluster_summaries: dict):
    cluster_ids = sorted(cluster_summaries.keys())
    persona_indices = list(range(len(PERSONA_LIBRARY)))

    if len(cluster_ids) > len(persona_indices):
        raise ValueError("More clusters than personas in PERSONA_LIBRARY.")

    best_total = -math.inf
    best_assignment = None

    for perm in itertools.permutations(persona_indices, len(cluster_ids)):
        total = 0.0
        current = {}
        for cluster_id, persona_idx in zip(cluster_ids, perm):
            persona = PERSONA_LIBRARY[persona_idx]
            summary = cluster_summaries[cluster_id]
            s = score_persona(summary, persona)
            total += s
            current[cluster_id] = {
                "persona": persona,
                "score": s,
            }

        if total > best_total:
            best_total = total
            best_assignment = current

    return best_assignment


def build_user_taste_summary(row: pd.Series):
    top_genres = safe_top_from_prefix(row, "genre_mean_", top_n=3)
    top_eras = safe_top_from_prefix(row, "era_", top_n=2)
    pop_pref = describe_popularity(row)
    behavior = describe_behavior(row)

    return {
        "user_top_genres": ", ".join(top_genres) if top_genres else "",
        "user_top_eras": ", ".join(top_eras) if top_eras else "",
        "user_popularity_pref": pop_pref,
        "user_behavior_profile": behavior,
    }


def main():
    if CLUSTER_PERSONAS_OUT.exists() and USER_DASHBOARD_OUT.exists():
        print("[07_dashboard_personas] Cache exists. Skipping persona mapping.")
        return

    features = pd.read_parquet(FEATURE_MATRIX_IN)
    cluster_assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)

    df = features.merge(
        cluster_assignments[["user", "cluster"]],
        on="user",
        how="left"
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [c for c in numeric_cols if c not in ["cluster"]]

    cluster_means = df.groupby("cluster")[numeric_feature_cols].mean()

    cluster_summaries = {}
    for cluster_id, row in cluster_means.iterrows():
        cluster_summaries[int(cluster_id)] = summarize_cluster(row)

    assignment = best_unique_persona_assignment(cluster_summaries)

    cluster_persona_rows = []
    for cluster_id in sorted(cluster_summaries.keys()):
        persona = assignment[cluster_id]["persona"]
        score = assignment[cluster_id]["score"]
        summary = cluster_summaries[cluster_id]

        cluster_persona_rows.append({
            "cluster": cluster_id,
            "persona_key": persona["persona_key"],
            "persona_name": persona["persona_name"],
            "short_label": persona["short_label"],
            "interpretation": persona["interpretation"],
            "example_movies": " | ".join(persona["example_movies"]),
            "top_genres": " | ".join(summary["top_genres"]),
            "top_eras": " | ".join(summary["top_eras"]),
            "popularity_pref": summary["popularity_pref"],
            "behavior_profile": summary["behavior_profile"],
            "persona_match_score": score,
        })

    cluster_personas = pd.DataFrame(cluster_persona_rows)

    user_summary_rows = []
    for _, row in df.iterrows():
        summary = build_user_taste_summary(row)
        user_summary_rows.append({
            "user": row["user"],
            "cluster": row["cluster"],
            "fsa": row["fsa"] if "fsa" in row.index else None,
            "region": row["region"] if "region" in row.index else None,
            "user_top_genres": summary["user_top_genres"],
            "user_top_eras": summary["user_top_eras"],
            "user_popularity_pref": summary["user_popularity_pref"],
            "user_behavior_profile": summary["user_behavior_profile"],
        })

    user_summary = pd.DataFrame(user_summary_rows)

    user_dashboard = user_summary.merge(
        cluster_personas,
        on="cluster",
        how="left"
    )

    user_dashboard.to_parquet(USER_DASHBOARD_OUT, index=False)
    cluster_personas.to_parquet(CLUSTER_PERSONAS_OUT, index=False)

    print("[07_dashboard_personas] Saved:", CLUSTER_PERSONAS_OUT)
    print("[07_dashboard_personas] Saved:", USER_DASHBOARD_OUT)


if __name__ == "__main__":
    main()