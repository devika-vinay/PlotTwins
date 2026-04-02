# narrative_utils.py
import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


CACHE_DIR = Path("data/cache")
USER_NARRATIVES_PATH = CACHE_DIR / "user_narratives.parquet"
MODEL_NAME = os.getenv("PLOTTWINS_NARRATIVE_MODEL", "gpt-5")


load_dotenv()


def safe_split_csv(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_split_pipe(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def format_era_label(raw):
    if pd.isna(raw):
        return ""
    s = str(raw).strip().replace("era_", "")
    try:
        year = int(float(s))
        return f"{year}s"
    except Exception:
        return s


def format_eras(values):
    return [format_era_label(v) for v in values if str(v).strip()]


def prettify_popularity(pop_value):
    mapping = {
        "pop_high": "blockbuster leaning",
        "pop_mid": "a mix of mainstream favorites and less obvious picks",
        "pop_low": "hidden-gem leaning",
    }
    return mapping.get(str(pop_value), str(pop_value))


def prettify_behavior(behavior):
    mapping = {
        "strongly positive": "a generous rater",
        "slightly positive": "fairly easy to win over",
        "neutral": "balanced in your ratings",
        "more critical": "selective with praise",
        "unknown": "still developing a clear rating pattern",
    }
    return mapping.get(str(behavior), str(behavior))


def load_narrative_cache():
    if USER_NARRATIVES_PATH.exists():
        return pd.read_parquet(USER_NARRATIVES_PATH)
    return pd.DataFrame(columns=[
        "user", "taste_headline", "taste_story", "people_story",
        "raw_llm_response", "model_used"
    ])


def save_narrative_cache(df: pd.DataFrame):
    USER_NARRATIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(USER_NARRATIVES_PATH, index=False)


def build_match_summary(user_matches: pd.DataFrame):
    if user_matches.empty:
        return {
            "match_names": [],
            "same_region_count": 0,
            "same_cluster_count": 0,
            "same_fsa_count": 0,
        }

    return {
        "match_names": user_matches["match_user"].astype(str).tolist(),
        "same_region_count": int(user_matches["same_region"].fillna(0).sum()) if "same_region" in user_matches.columns else 0,
        "same_cluster_count": int(user_matches["same_cluster"].fillna(0).sum()) if "same_cluster" in user_matches.columns else 0,
        "same_fsa_count": int(user_matches["same_fsa"].fillna(0).sum()) if "same_fsa" in user_matches.columns else 0,
    }


def build_prompt_payload(user_row: pd.Series, user_matches: pd.DataFrame):
    return {
        "user": str(user_row.get("user", "")),
        "persona_name": str(user_row.get("persona_name", "")),
        "short_label": str(user_row.get("short_label", "")),
        "interpretation": str(user_row.get("interpretation", "")),
        "user_top_genres": safe_split_csv(user_row.get("user_top_genres", ""))[:5],
        "user_top_eras": format_eras(safe_split_csv(user_row.get("user_top_eras", "")))[:3],
        "user_popularity_pref": prettify_popularity(user_row.get("user_popularity_pref", "")),
        "user_behavior_profile": prettify_behavior(user_row.get("user_behavior_profile", "")),
        "cluster_top_genres": safe_split_pipe(user_row.get("top_genres", ""))[:5],
        "cluster_top_eras": format_eras(safe_split_pipe(user_row.get("top_eras", "")))[:3],
        "example_movies": safe_split_pipe(user_row.get("example_movies", ""))[:5],
        "match_summary": build_match_summary(user_matches),
    }


def extract_response_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    texts = []
    try:
        for item in response.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                if getattr(block, "type", None) == "output_text":
                    texts.append(block.text)
    except Exception:
        pass
    return "\n".join(texts).strip()


def generate_narrative_for_user(user_row: pd.Series, user_matches: pd.DataFrame):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

    client = OpenAI(api_key=api_key)
    payload = build_prompt_payload(user_row, user_matches)

    developer_prompt = """
You are writing user-facing copy for a movie taste app called PlotTwins.

Rules:
1. Stay fully grounded in the provided data.
2. Do not invent genres, eras, match reasons, or movies.
3. Write in a warm, polished, Spotify Wrapped–style tone.
4. Avoid robotic or repetitive phrasing.
5. Explain ideas naturally for a non-technical user.
6. Return only valid JSON with exactly these keys:
   - taste_headline
   - taste_story
   - people_story

Style:
- taste_headline: 5 to 10 words
- taste_story: 70 to 120 words
- people_story: 18 to 35 words
"""

    user_prompt = f"""
Here is the grounded user context as JSON:

{json.dumps(payload, ensure_ascii=False, indent=2)}

Write the JSON output now.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = extract_response_text(response)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "taste_headline": str(user_row.get("short_label", "Your movie identity")),
            "taste_story": str(user_row.get("interpretation", "")),
            "people_story": "These are the users whose taste patterns align most closely with yours.",
        }

    return {
        "user": str(user_row.get("user", "")),
        "taste_headline": str(parsed.get("taste_headline", "")).strip(),
        "taste_story": str(parsed.get("taste_story", "")).strip(),
        "people_story": str(parsed.get("people_story", "")).strip(),
        "raw_llm_response": raw_text,
        "model_used": MODEL_NAME,
    }


def get_or_create_narrative(user_row: pd.Series, user_matches: pd.DataFrame):
    cache_df = load_narrative_cache()
    user_id = str(user_row.get("user", ""))

    existing = cache_df[cache_df["user"].astype(str) == user_id]
    if not existing.empty:
        return existing.iloc[0].to_dict(), False

    narrative = generate_narrative_for_user(user_row, user_matches)
    updated = pd.concat([cache_df, pd.DataFrame([narrative])], ignore_index=True)
    updated = updated.drop_duplicates(subset=["user"], keep="last")
    save_narrative_cache(updated)

    return narrative, True