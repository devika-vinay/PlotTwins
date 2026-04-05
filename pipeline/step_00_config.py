from pathlib import Path

# Get the root directory of the project (two levels up from this file)
ROOT = Path(__file__).resolve().parents[1]

# Define key data directories relative to the project root
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

# Create the cache and output directories if they don't already exist
# parents=True ensures any missing parent folders are also created
# exist_ok=True prevents errors if the folder already exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source locations
# URL to the raw (uncleaned) movies dataset stored in cloud storage
MOVIES_CSV_PATH = "https://g2analyticscapstone.blob.core.windows.net/plottwinsfiles/Uncleaned_%20movie%20dataset.csv?sp=r&st=2026-03-10T18:30:07Z&se=2026-08-01T02:45:07Z&spr=https&sv=2024-11-04&sr=b&sig=uCDpnYVSpz6c4eeJmNV46Td6OLr%2Fil6v%2Bjowc%2FqcLV4%3D"

# Path to ratings dataset stored in a Hugging Face dataset repository (Parquet format)
RATINGS_PARQUET_PATH = r"hf://datasets/Shree2428/letterboxd-10m-movies-ratings-2025/train.parquet"
