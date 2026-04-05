# 🎬 PlotTwins: Rebuilding Social Movie Experiences

## 📌 Overview

Movie watching has increasingly become an isolated, streaming-first activity, leading to declining theatre engagement for companies like Cineplex.

PlotTwins aims to reverse this trend by:

> Grouping users with similar movie tastes and enabling shared, in-theatre experiences.

Using large-scale Letterboxd-style data (~10M+ ratings), we built a data pipeline + user profiling system that:

* Identifies users with similar preferences
* Groups them into meaningful communities
* Enables event-based recommendations (e.g., themed screenings)
* Generates human-readable narratives describing user and cluster taste profiles
* Reintroduces movie watching as a social activity

**Try it live:**  
- [User Dashboard](https://plot-twins.vercel.app/)  
- [Business Insights Layer](https://plot-twins.vercel.app/business)

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/devika-vinay/PlotTwins.git
cd plot-twins
```

---

### 2. Run pipeline (in a terminal)

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

### What happens:


* Creates virtual environment
* Installs dependencies
* Runs full pipeline
* Caches intermediate results
* Builds user-level taste profiles and clusters
* Generates narratives for user and cluster personas

---

## 🤝 Contributor Guide

### Workflow

1. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

2. Make changes (add new files in the existing format)

3. Add any new steps to run_pipeline.sh

4. Commit clearly:

```bash
git commit -m "Add user profile feature aggregation"
```

5. Push + open PR
```bash
git push origin feature/your-branch-name
```

---

### Guidelines

* Keep pipeline steps modular
* Cache intermediate outputs
* Avoid hardcoding paths (use `config.py`)
* Do NOT commit:

  * `data/cache/`
  * `venv/`
  * `__pycache__/`

---

### Note

If you would like to run individual files:
1. Ensure any required cache data has been created from prior steps
2. Create a virtual environment by running these commands on your terminal
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python ./<your file name>
```

---

