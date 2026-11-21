"""Momentum index for quarterbacks with 20-49 starts (e.g., Bo Nix), so they can be
searched on the page. Not part of the funnel statistics, which use 50+ starts."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploratory import unit_index

ROOT = Path(__file__).resolve().parents[1]
p = pd.read_parquet(ROOT / "data" / "clean" / "nfl_plays_model.parquet")
q = p[p.passer_player_name.notna()].groupby(["game_id", "posteam", "passer_player_name"]).size().rename("n").reset_index()
q = q.sort_values("n").groupby(["game_id", "posteam"]).tail(1)[["game_id", "posteam", "passer_player_name"]]
pp = p.merge(q, on=["game_id", "posteam"], how="inner", suffixes=("_play", ""))
games = pp.groupby("passer_player_name").game_id.nunique()
keep = games[(games >= 20) & (games < 50)].index
t = unit_index(pp[pp.passer_player_name.isin(keep)], ["passer_player_name"], min_pairs=300)
t["games"] = t.passer_player_name.map(games)
(ROOT / "results" / "qbs_small.json").write_text(t.to_json(orient="records"))
print(len(t)); print(t[t.passer_player_name.isin(["B.Nix", "C.Williams", "J.Daniels", "D.Maye", "C.Stroud"])][["passer_player_name", "games", "excess", "z"]])
