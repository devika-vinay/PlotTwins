from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source locations
MOVIES_CSV_PATH = r"C:\Users\Devika Vinay\OneDrive\Desktop\Winter 26\Capstone\PlotTwins\data\raw\movie_data.csv"
RATINGS_PARQUET_PATH = r"hf://datasets/Shree2428/letterboxd-10m-movies-ratings-2025/train.parquet"

