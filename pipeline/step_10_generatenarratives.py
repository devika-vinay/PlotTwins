import os
import json
import time
import pandas as pd
from pipeline.step_00_config import CACHE_DIR
from dotenv import load_dotenv
from openai import OpenAI
# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()  # loads .env into environment

api_key = os.getenv("OPENAI_API_KEY")
# ── Input and output artifacts ────────────────────────────────────────────
USER_DASHBOARD_IN = CACHE_DIR / "user_dashboard.parquet" # per-user dashboard from step 09
USER_MATCHES_IN = CACHE_DIR / "user_matches.parquet" # matched users info

USER_NARRATIVES_OUT = CACHE_DIR / "user_narratives.parquet" # output with generated narratives
# ── Config ───────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("PLOTTWINS_NARRATIVE_MODEL", "gpt-5") # default model
SLEEP_BETWEEN_CALLS = 0.3 # rate-limit between API calls

# ── Helper functions ──────────────────────────────────────────────────────
def safe_split_csv(value):
    """Split a comma-separated string into a list, ignoring blanks or NaN"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_split_pipe(value):
    """Split a pipe-separated string into a list, ignoring blanks or NaN"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def format_era_label(raw):
    """Convert raw era labels like 'era_1990.0' → '1990s'"""
    if pd.isna(raw):
        return ""
    s = str(raw).strip().replace("era_", "")
    try:
        year = int(float(s))
        return f"{year}s"
    except Exception:
        return s


def format_eras(values):
    """Apply era formatting to a list of values"""
    return [format_era_label(v) for v in values if str(v).strip()]


def prettify_popularity(pop_value):
    """Map popularity tier codes to user-friendly strings"""
    mapping = {
        "pop_high": "blockbuster leaning",
        "pop_mid": "a mix of mainstream favorites and less obvious picks",
        "pop_low": "hidden-gem leaning",
    }
    return mapping.get(str(pop_value), str(pop_value))


def prettify_behavior(behavior):
    """Map cluster/user behavior codes to user-friendly strings"""
    mapping = {
        "strongly positive": "a generous rater",
        "slightly positive": "fairly easy to win over",
        "neutral": "balanced in your ratings",
        "more critical": "selective with praise",
        "unknown": "still developing a clear rating pattern",
    }
    return mapping.get(str(behavior), str(behavior))


def build_match_summary(user_matches: pd.DataFrame):
    """Summarize the user's matches by counts and names"""
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
    """Construct a JSON payload with user + cluster info for LLM prompt"""
    user_genres = safe_split_csv(user_row.get("user_top_genres", ""))
    user_eras = format_eras(safe_split_csv(user_row.get("user_top_eras", "")))
    cluster_genres = safe_split_pipe(user_row.get("top_genres", ""))
    cluster_eras = format_eras(safe_split_pipe(user_row.get("top_eras", "")))
    example_movies = safe_split_pipe(user_row.get("example_movies", ""))

    payload = {
        "user": str(user_row.get("user", "")),
        "persona_name": str(user_row.get("persona_name", "")),
        "short_label": str(user_row.get("short_label", "")),
        "interpretation": str(user_row.get("interpretation", "")),
        "user_top_genres": user_genres[:5],
        "user_top_eras": user_eras[:3],
        "user_popularity_pref": prettify_popularity(user_row.get("user_popularity_pref", "")),
        "user_behavior_profile": prettify_behavior(user_row.get("user_behavior_profile", "")),
        "cluster_top_genres": cluster_genres[:5],
        "cluster_top_eras": cluster_eras[:3],
        "cluster_popularity_pref": str(user_row.get("popularity_pref", "")),
        "cluster_behavior_profile": str(user_row.get("behavior_profile", "")),
        "example_movies": example_movies[:5],
        "match_summary": build_match_summary(user_matches),
    }
    return payload


def extract_response_text(response):
    """Extract text from OpenAI response object, with fallback"""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Fallback traversal
    try:
        texts = []
        for item in response.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                if getattr(block, "type", None) == "output_text":
                    texts.append(block.text)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def generate_narrative(client: OpenAI, payload: dict):
    """Call OpenAI to generate a PlotTwins narrative for a single user"""
    developer_prompt = """
You are writing user-facing copy for a movie taste app called PlotTwins.

You must follow these rules:
1. Stay fully grounded in the provided data.
2. Do not invent genres, eras, match reasons, or movies that are not in the input.
3. Write in a warm, polished, Spotify Wrapped–style tone.
4. Avoid sounding robotic, repetitive, or overly technical.
5. Explain terms naturally. For example:
   - "blockbuster leaning" should feel like "you connect more with widely loved or culturally visible films"
   - "balanced ratings" should feel like "you appreciate strong films but do not hand out praise too easily"
6. Keep the output concise and UI-friendly.
7. Return only valid JSON with exactly these keys:
   - taste_headline
   - taste_story
   - people_story

Style guidance:
- taste_headline: 5 to 10 words, punchy and human
- taste_story: one paragraph, 70 to 120 words
- people_story: one sentence, 18 to 35 words

Do not use markdown.
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
        # fallback: preserve something usable
        parsed = {
            "taste_headline": payload.get("short_label", "Your movie identity"),
            "taste_story": payload.get("interpretation", ""),
            "people_story": "These are the users whose taste patterns align most closely with yours.",
            "raw_response": raw_text,
        }

    return {
        "taste_headline": str(parsed.get("taste_headline", "")).strip(),
        "taste_story": str(parsed.get("taste_story", "")).strip(),
        "people_story": str(parsed.get("people_story", "")).strip(),
        "raw_llm_response": raw_text,
    }

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    # Validate inputs
    if not USER_DASHBOARD_IN.exists():
        raise FileNotFoundError(f"[10_generate_narratives] Missing input: {USER_DASHBOARD_IN}")

    if not USER_MATCHES_IN.exists():
        raise FileNotFoundError(f"[10_generate_narratives] Missing input: {USER_MATCHES_IN}")

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "[10_generate_narratives] OPENAI_API_KEY is not set. "
            "Set it in your environment before running this step."
        )
    # Load user data and matches
    dashboard = pd.read_parquet(USER_DASHBOARD_IN)
    matches = pd.read_parquet(USER_MATCHES_IN)
    # Handle incremental processing if some narratives exist
    existing = pd.DataFrame()
    if USER_NARRATIVES_OUT.exists():
        existing = pd.read_parquet(USER_NARRATIVES_OUT)
        print("[10_generate_narratives] Existing cache found. Will only generate missing users.")

    done_users = set(existing["user"].astype(str).tolist()) if not existing.empty else set()

    client = OpenAI(api_key=api_key)

    rows = []
    # Iterate over users and generate narratives
    for _, user_row in dashboard.iterrows():
        user_id = str(user_row["user"])

        if user_id in done_users:
            continue

        user_matches = matches[matches["user"].astype(str) == user_id].copy()
        payload = build_prompt_payload(user_row, user_matches)

        try:
            narrative = generate_narrative(client, payload)
            rows.append({
                "user": user_id,
                "taste_headline": narrative["taste_headline"],
                "taste_story": narrative["taste_story"],
                "people_story": narrative["people_story"],
                "raw_llm_response": narrative["raw_llm_response"],
                "model_used": MODEL_NAME,
            })
            print(f"[10_generate_narratives] Generated narrative for user {user_id}")
        except Exception as e:
            # fallback in case of API failure
            rows.append({
                "user": user_id,
                "taste_headline": str(user_row.get("short_label", "Your movie identity")),
                "taste_story": str(user_row.get("interpretation", "")),
                "people_story": "These are the users whose taste patterns align most closely with yours.",
                "raw_llm_response": f"ERROR: {e}",
                "model_used": MODEL_NAME,
            })
            print(f"[10_generate_narratives] Failed for user {user_id}: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)
    # Combine with existing cache and remove duplicates
    new_df = pd.DataFrame(rows)

    if existing.empty:
        final_df = new_df
    elif new_df.empty:
        final_df = existing
    else:
        final_df = pd.concat([existing, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["user"], keep="last")
    # Save final output
    final_df.to_parquet(USER_NARRATIVES_OUT, index=False)
    print("[10_generate_narratives] Saved:", USER_NARRATIVES_OUT)


if __name__ == "__main__":
    main()