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

# ----------------------------
# 4. Run optional pipeline steps
# ----------------------------
# Optional EDA step - keep commented unless running some tests on any data
# python -m pipeline.step_00_edacheck

# Generate narratives for every single user beforehand
# python -m pipeline.step_10_generatenarratives

# Uncomment if cold-start prediction is needed
# python -m pipeline.step_11_coldstart
# python -m pipeline.step_12_predict_new_user

# ----------------------------
# 5. Start Streamlit app
# ----------------------------
streamlit run app.py

echo "Pipeline complete!"