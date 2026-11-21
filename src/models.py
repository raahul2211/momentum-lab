"""Expected-outcome models ("how hard was this play / drive?").

The momentum literature that finds a hot hand does so after modeling difficulty
(Bocskocsky, Ezekowitz & Stein 2014 for shot difficulty). Football's version of shot
difficulty is the situation plus who is on the field. These models use ONLY:
  - the situation at the snap (down, distance, field, clock, score, timeouts),
  - pregame team strength (betting spread/total, prior-games form),
  - environment (home, dome, temp, wind, season).
They never see anything that happened earlier in the same game, so they cannot
soak up momentum. Predictions are cross-fitted: each season is predicted by a
model that never saw that season.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

PLAY_FEATURES = [
    "down", "ydstogo", "yardline_100", "goal_to_go", "qtr", "half_seconds_remaining",
    "game_seconds_remaining", "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "home", "pos_spread", "pos_implied", "def_implied",
    "total_line", "dome", "temp", "wind", "off_sr_pre", "def_sr_pre", "season",
    "first_play_of_drive", "pass",
]
DRIVE_FEATURES = [
    "start_yardline_100", "start_half_secs", "start_game_secs", "qtr", "start_score_diff",
    "pos_to", "def_to", "home", "pos_spread", "pos_implied", "def_implied", "total_line",
    "dome", "off_sr_pre", "def_sr_pre", "season",
]


def season_folds(seasons, k=6):
    u = np.sort(np.unique(seasons))
    return {s: i % k for i, s in enumerate(u)}


def crossfit(df, features, target, kind, k=6, seed=0):
    fold_of = season_folds(df.season.values, k)
    folds = df.season.map(fold_of).values
    pred = np.zeros(len(df))
    X = df[features].astype(float).values
    y = df[target].values
    for f in range(k):
        tr, te = folds != f, folds == f
        if kind == "clf":
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
                                               min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=32,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return pred


def fit_nfl(play_features=PLAY_FEATURES, out_suffix=""):
    p = pd.read_parquet(CLEAN / "nfl_plays.parquet")
    p["p_hat"] = crossfit(p, play_features, "success", "clf")
    p["r"] = p.success - p.p_hat
    # EPA version (secondary outcome)
    p["epa_hat"] = crossfit(p, play_features, "epa", "reg")
    p["r_epa"] = p.epa - p.epa_hat
    # Pre-snap stakes: expected |win-probability swing| given the situation (a leverage index).
    lev_feats = [f for f in play_features if f != "pass"]
    p["lev_pre"] = crossfit(p.assign(lev=p.lev.fillna(0)), lev_feats, "lev", "reg")
    p.to_parquet(CLEAN / f"nfl_plays_model{out_suffix}.parquet", index=False)

    d = pd.read_parquet(CLEAN / "nfl_drives.parquet")
    d["pts_hat"] = crossfit(d, DRIVE_FEATURES, "points", "reg")
    d["poe"] = d.points - d.pts_hat
    # Drive-level "over expected" built as SUMS of play residuals. Under no momentum each
    # play residual has mean zero given its situation, so the sum is a martingale and has
    # mean zero whatever the drive length (optional stopping). A per-drive MEAN would not:
    # short-field drives end fast when they work, which biases the mean.
    key = p.game_id + "_" + p.fixed_drive.astype("Int64").astype(str)
    sums = p.assign(drive_key=key).groupby("drive_key").agg(sr_sum=("r", "sum"), epa_oe=("r_epa", "sum"))
    d["sr_sum"] = d.drive_key.map(sums.sr_sum)
    d["epa_oe"] = d.drive_key.map(sums.epa_oe)
    d.to_parquet(CLEAN / f"nfl_drives_model{out_suffix}.parquet", index=False)
    return p, d


CFB_PLAY = ["down", "ydstogo", "yardline_100", "goal_to_go", "qtr", "half_seconds_remaining", "game_seconds_remaining",
            "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "pos_spread",
            "total_line", "off_sr_pre", "def_sr_pre", "season", "first_play_of_drive", "pass"]
CFB_DRIVE = ["start_yardline_100", "start_half_secs", "start_game_secs", "qtr", "start_score_diff", "pos_to", "def_to",
             "home", "pos_spread", "total_line", "off_sr_pre", "def_sr_pre", "season"]


def fit_cfb():
    p = pd.read_parquet(CLEAN / "cfb_plays.parquet")
    p["p_hat"] = crossfit(p, CFB_PLAY, "success", "clf", k=4)
    p["r"] = p.success - p.p_hat
    p["epa_hat"] = crossfit(p, CFB_PLAY, "epa", "reg", k=4)
    p["r_epa"] = p.epa - p.epa_hat
    p["lev_pre"] = crossfit(p.assign(lev=p.lev.fillna(0)), [f for f in CFB_PLAY if f != "pass"], "lev", "reg", k=4)
    p.to_parquet(CLEAN / "cfb_plays_model.parquet", index=False)
    d = pd.read_parquet(CLEAN / "cfb_drives.parquet")
    d["pts_hat"] = crossfit(d, CFB_DRIVE, "points", "reg", k=4)
    d["poe"] = d.points - d.pts_hat
    key = p.game_id.astype(str) + "_" + p.fixed_drive.astype(str)
    sums = p.assign(drive_key=key).groupby("drive_key").agg(sr_sum=("r", "sum"), epa_oe=("r_epa", "sum"))
    d["sr_sum"] = d.drive_key.map(sums.sr_sum)
    d["epa_oe"] = d.drive_key.map(sums.epa_oe)
    d.to_parquet(CLEAN / "cfb_drives_model.parquet", index=False)
    return p, d


if __name__ == "__main__":
    import time
    t = time.time()
    import sys
    p, d = fit_cfb() if "cfb" in sys.argv else fit_nfl()
    print("plays", len(p), "mean r", p.r.mean().round(4), "drives", len(d), "mean poe", d.poe.mean().round(3),
          f"{time.time()-t:.0f}s")

