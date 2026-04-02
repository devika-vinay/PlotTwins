#!/usr/bin/env bash
set -euo pipefail

echo "Starting pipeline..."

# ----------------------------
# 1. Create virtual environment (if not exists)
# ----------------------------
if [ ! -d "venv" ]; then
  echo "Creating and activating virtual environment..."
  python -m venv venv
fi
source venv/Scripts/activate

# ----------------------------
# 2. Install requirements (only if needed)
# ----------------------------
echo "Installing dependencies..."
pip install -r requirements.txt

# ----------------------------
# 3. Run pipeline steps
# ----------------------------
echo "Running pipeline..."
python -m pipeline.step_01_load
python -m pipeline.step_02_transform
python -m pipeline.step_03_userprofile
python -m pipeline.step_04_featurematrix
python -m pipeline.step_05_clusterusers
python -m pipeline.step_06_matchusers
python -m pipeline.step_07_cluster_kb
python -m pipeline.step_08_movies_kb
python -m pipeline.step_09_dashboardpersonas
# Step 10 is only for extreme cases if we want to generate narratives for every single user beforehand
streamlit run app.py

# Optional EDA step - keep commented unless running some tests on any data
# python -m pipeline.step_00_edacheck


echo "Pipeline complete!"