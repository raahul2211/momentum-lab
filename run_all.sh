#!/usr/bin/env bash
# Full pipeline: data -> tables -> models -> tests -> exploratory -> page data -> notebook
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/nfl data/raw/cfb
for y in $(seq 1999 2026); do
  f=data/raw/nfl/pbp_$y.parquet
  [ -f "$f" ] || curl -sL -o "$f" "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_$y.parquet"
done
[ -f data/raw/nfl/games.csv ] || curl -sL -o data/raw/nfl/games.csv https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
if [ ! -f data/raw/cfb/play_by_play_2021.parquet ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/sportsdataverse/cfbfastR-data /tmp/cfbfastR-data
  cp /tmp/cfbfastR-data/pbp/parquet/play_by_play_20{14..21}.parquet data/raw/cfb/
fi
python src/build_nfl.py
python src/build_cfb.py
python src/models.py
python src/models.py cfb
python -c "import sys; sys.path.insert(0,'src'); import pandas as pd; from analysis import prep_plays; from simulate import run; p=pd.read_parquet('data/clean/nfl_plays_model.parquet'); run(prep_plays(p, list(range(2009,2019))), n_sims=30, n_perm=150).to_parquet('data/clean/sim_results.parquet')"
for s in "nfl discovery" "nfl holdout" "nfl fresh" "cfb discovery" "cfb holdout"; do python src/run_confirmatory.py $s; done
python src/exploratory.py
python src/stress.py
python src/qb_small.py
python src/build_page_data.py
jupyter nbconvert --to notebook --execute --inplace notebooks/momentum.ipynb
