"""College (FBS) tables from cfbfastR play-by-play, built to the same schema as the NFL
tables so every test runs unchanged. Seasons 2014-2021 (betting lines available).

Known limits vs NFL: no flag for aborted snaps or fumbles out of bounds, drive points are
taken from the drive result (TD = 7), and coverage is FBS games with a posted line.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cfb"
CLEAN = ROOT / "data" / "clean"

COLS = ["game_id", "season", "seasonType", "week", "id", "sequenceNumber", "game_play_number", "pos_team", "def_pos_team",
        "is_home", "drive.id", "drive.result", "period", "start.down", "start.distance", "start.yardsToEndzone",
        "start.TimeSecsRem", "start.adj_TimeSecsRem", "pos_score_diff_start", "start.posTeamTimeouts",
        "start.defPosTeamTimeouts", "EPA", "scrimmage_play", "rush", "pass", "sack", "int", "fumble_vec",
        "fumble_lost", "wp_before", "wpa", "start.pos_team_spread", "overUnder", "gameSpreadAvailable", "text",
        "homeTeamName", "awayTeamName"]

PTS = {"TD": 7, "FG": 3, "SF": -2, "INT TD": -7, "FUMBLE TD": -7, "FUMBLE RETURN TD": -7, "PUNT TD": -7,
       "PUNT RETURN TD": -7, "MISSED FG TD": -7, "DOWNS TD": -7, "FG TD": -7}


def build(seasons=range(2014, 2022)):
    d = pd.concat([pd.read_parquet(RAW / f"play_by_play_{s}.parquet", columns=COLS) for s in seasons], ignore_index=True)
    d = d[d.gameSpreadAvailable == True].copy()  # noqa: E712
    # ESPN files are stored newest-first for many games; game_play_number follows file order.
    # sequenceNumber is ESPN's play id and runs in game order, so we order by it.
    d["seqn"] = pd.to_numeric(d.sequenceNumber, errors="coerce")
    d = d.dropna(subset=["seqn"]).drop_duplicates(["game_id", "seqn"])
    d = d.sort_values(["game_id", "seqn"]).reset_index(drop=True)
    d["game_play_number"] = d.groupby("game_id").cumcount() + 1
    d = d.rename(columns={
        "pos_team": "posteam", "def_pos_team": "defteam", "period": "qtr", "start.down": "down",
        "start.distance": "ydstogo", "start.yardsToEndzone": "yardline_100", "start.TimeSecsRem": "half_seconds_remaining",
        "start.adj_TimeSecsRem": "game_seconds_remaining", "pos_score_diff_start": "score_differential",
        "start.posTeamTimeouts": "posteam_timeouts_remaining", "start.defPosTeamTimeouts": "defteam_timeouts_remaining",
        "EPA": "epa", "wp_before": "wp", "start.pos_team_spread": "pos_spread", "overUnder": "total_line",
        "seasonType": "season_type"})
    d["home"] = d.is_home.astype(float)
    d["half"] = np.where(d.qtr <= 2, 1, 2)
    # drive order inside each game
    first_play = d.groupby("drive.id").game_play_number.min()
    order = first_play.reset_index().merge(d[["drive.id", "game_id"]].drop_duplicates(), on="drive.id")
    order = order.sort_values(["game_id", "game_play_number"])
    order["fixed_drive"] = order.groupby("game_id").cumcount() + 1
    d = d.merge(order[["drive.id", "fixed_drive"]], on="drive.id", how="left")

    scr = (d.scrimmage_play == 1) & d.down.between(1, 4) & d.epa.notna() & (d.qtr <= 4) & d.posteam.notna()
    # Same snap logged twice (e.g. "run for 19 yds for a TD" then "19 Yd Run (Kick)"), or a
    # nullified play followed by the replay of the down: keep the last row of each snap.
    snap = ["game_id", "posteam", "qtr", "half_seconds_remaining", "down", "ydstogo", "yardline_100"]
    dup_snap = scr & d[snap].eq(d[snap].shift(-1)).all(axis=1) & scr.shift(-1, fill_value=False)
    scr = scr & ~dup_snap
    d["is_scrimmage"] = scr.astype(int)
    d["success"] = (d.epa > 0).astype(int)
    d["pass"] = d["pass"].astype(float)
    d["goal_to_go"] = (d.ydstogo >= d.yardline_100).astype(int)

    # prior-games form, same recipe as NFL
    s = d[d.is_scrimmage == 1]
    wk = d.groupby("game_id").agg(season=("season", "first"), week=("week", "first"), st=("season_type", "first")).reset_index()
    league = s.success.mean()
    for side, col in [("off", "posteam"), ("def", "defteam")]:
        t = s.groupby(["game_id", col]).success.mean().rename("v").reset_index().merge(wk, on="game_id")
        t = t.sort_values([col, "season", "st", "week"])
        t[f"{side}_sr_pre"] = t.groupby(col).v.transform(lambda x: x.shift(1).ewm(halflife=8, min_periods=1).mean()).fillna(league)
        d = d.merge(t[["game_id", col, f"{side}_sr_pre"]], on=["game_id", col], how="left")
        d[f"{side}_sr_pre"] = d[f"{side}_sr_pre"].fillna(league)

    p = d[d.is_scrimmage == 1].copy()
    p["first_play_of_drive"] = (p.groupby("drive.id").cumcount() == 0).astype(int)
    p["lev"] = p.wpa.abs()
    p["seq"] = p.groupby(["game_id", "posteam"]).cumcount()
    p["play_id"] = p.game_play_number
    p["dome"] = 0
    keep = ["game_id", "season", "season_type", "week", "posteam", "defteam", "play_id", "fixed_drive", "qtr", "half",
            "down", "ydstogo", "yardline_100", "goal_to_go", "game_seconds_remaining", "half_seconds_remaining",
            "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "pass", "epa", "success",
            "wp", "wpa", "lev", "home", "pos_spread", "total_line", "dome", "off_sr_pre", "def_sr_pre",
            "first_play_of_drive", "seq", "homeTeamName", "awayTeamName", "text"]
    p = p[keep].reset_index(drop=True)
    p.to_parquet(CLEAN / "cfb_plays.parquet", index=False)

    agg = p.groupby(["game_id", "fixed_drive"]).agg(
        season=("season", "first"), week=("week", "first"), posteam=("posteam", "first"), defteam=("defteam", "first"),
        qtr=("qtr", "first"), end_qtr=("qtr", "last"), half=("half", "first"),
        start_yardline_100=("yardline_100", "first"), start_half_secs=("half_seconds_remaining", "first"),
        end_half_secs=("half_seconds_remaining", "last"), start_game_secs=("game_seconds_remaining", "first"),
        start_score_diff=("score_differential", "first"), pos_to=("posteam_timeouts_remaining", "first"),
        def_to=("defteam_timeouts_remaining", "first"), n_plays=("play_id", "size"), home=("home", "first"),
        pos_spread=("pos_spread", "first"), total_line=("total_line", "first"), off_sr_pre=("off_sr_pre", "first"),
        def_sr_pre=("def_sr_pre", "first"), dome=("dome", "first")).reset_index()
    res = d.groupby(["game_id", "fixed_drive"])["drive.result"].first()
    agg["result"] = res.reindex(pd.MultiIndex.from_frame(agg[["game_id", "fixed_drive"]])).values
    agg["points"] = agg.result.map(PTS).fillna(0)
    agg["duration"] = agg.start_half_secs - agg.end_half_secs
    agg["drive_key"] = agg.game_id.astype(str) + "_" + agg.fixed_drive.astype(str)
    agg = agg[agg.half == np.where(agg.end_qtr <= 2, 1, 2)].sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    agg.to_parquet(CLEAN / "cfb_drives.parquet", index=False)

    f = d[(d.is_scrimmage == 1) & (d.fumble_vec == 1)].copy()
    f["lost"] = (f.fumble_lost == 1).astype(int)
    f["ftype"] = np.select([f.sack == 1, f["pass"] == 1, f.rush == 1], ["sack", "reception", "rush"], "other")
    f["fumble_forced"] = 1.0  # not recorded in cfbfastR; stratify by type only
    f["return_touchdown"] = f["drive.result"].isin(["FUMBLE TD", "FUMBLE RETURN TD"]).astype(int)
    f = f[["game_id", "season", "posteam", "defteam", "fixed_drive", "qtr", "half", "down", "ydstogo", "yardline_100",
           "half_seconds_remaining", "game_seconds_remaining", "score_differential", "wp", "home", "pos_spread",
           "fumble_forced", "lost", "ftype", "return_touchdown", "text"]]
    f.to_parquet(CLEAN / "cfb_fumbles.parquet", index=False)
    print(f"cfb plays {len(p):,} drives {len(agg):,} fumbles {len(f):,} games {p.game_id.nunique():,}")


if __name__ == "__main__":
    build()
