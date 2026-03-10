from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source locations
MOVIES_CSV_PATH = "https://g2analyticscapstone.blob.core.windows.net/plottwinsfiles/Uncleaned_%20movie%20dataset.csv?sp=r&st=2026-03-10T18:30:07Z&se=2026-08-01T02:45:07Z&spr=https&sv=2024-11-04&sr=b&sig=uCDpnYVSpz6c4eeJmNV46Td6OLr%2Fil6v%2Bjowc%2FqcLV4%3D"
RATINGS_PARQUET_PATH = r"hf://datasets/Shree2428/letterboxd-10m-movies-ratings-2025/train.parquet"
