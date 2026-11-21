"""Build clean NFL play and drive tables from nflverse play-by-play.

Outputs (data/clean/):
  nfl_plays.parquet   one row per offensive scrimmage play (pass/run), quarters 1-4
  nfl_drives.parquet  one row per offensive drive with >=1 scrimmage play, quarters 1-4
  nfl_fumbles.parquet one row per live-ball scrimmage fumble (no aborted snaps, not out of bounds)

Nothing here looks at sequences of outcomes. It only cleans and builds pregame features.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nfl"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

COLS = """game_id season season_type week home_team away_team posteam defteam play_id fixed_drive
fixed_drive_result qtr down ydstogo yardline_100 goal_to_go game_seconds_remaining half_seconds_remaining
score_differential posteam_timeouts_remaining defteam_timeouts_remaining play_type pass rush qb_kneel
qb_spike epa wp wpa vegas_wp spread_line total_line roof temp wind fumble fumble_lost fumble_out_of_bounds
fumble_recovery_1_team fumble_forced aborted_play sack interception total_home_score total_away_score
two_point_attempt play_deleted touchdown return_touchdown td_team desc order_sequence
fourth_down_converted timeout timeout_team home_score away_score passer_player_id passer_player_name""".split()


def load_raw(seasons):
    frames = []
    for s in seasons:
        f = RAW / f"pbp_{s}.parquet"
        if not f.exists():
            continue
        frames.append(pd.read_parquet(f, columns=COLS))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.season_type.isin(["REG", "POST"])].copy()
    # Stable within-game order.
    d = d.sort_values(["game_id", "order_sequence", "play_id"]).reset_index(drop=True)
    return d


def add_pregame_strength(d):
    """Ex-ante team strength that never uses the current game.

    1. Market: spread and total -> implied points for the offense and the defense.
    2. Form: exponentially weighted success rate on offense and allowed on defense,
       over prior games only (shifted), half-life 8 games, carried across seasons.
    """
    home = d.posteam == d.home_team
    d["home"] = home.astype(int)
    # nflverse spread_line: points the home team is favored by.
    d["pos_spread"] = np.where(home, d.spread_line, -d.spread_line)
    d["pos_implied"] = d.total_line / 2 + d.pos_spread / 2
    d["def_implied"] = d.total_line / 2 - d.pos_spread / 2

    scr = d[d.is_scrimmage == 1]
    games = d.groupby("game_id").agg(season=("season", "first"), week=("week", "first")).reset_index()
    off = scr.groupby(["game_id", "posteam"]).success.mean().rename("off_sr").reset_index()
    dfn = scr.groupby(["game_id", "defteam"]).success.mean().rename("def_sr").reset_index()
    off = off.merge(games, on="game_id")
    dfn = dfn.merge(games, on="game_id")
    league = scr.success.mean()

    def prior_ewm(t, team_col, val):
        t = t.sort_values([team_col, "season", "week"]).copy()
        t[val + "_pre"] = (
            t.groupby(team_col)[val].transform(lambda x: x.shift(1).ewm(halflife=8, min_periods=1).mean())
        )
        t[val + "_pre"] = t[val + "_pre"].fillna(league)
        return t[["game_id", team_col, val + "_pre"]]

    d = d.merge(prior_ewm(off, "posteam", "off_sr"), on=["game_id", "posteam"], how="left")
    d = d.merge(prior_ewm(dfn, "defteam", "def_sr"), on=["game_id", "defteam"], how="left")
    d["off_sr_pre"] = d.off_sr_pre.fillna(league)
    d["def_sr_pre"] = d.def_sr_pre.fillna(league)
    return d


def build(seasons=range(1999, 2027)):
    d = load_raw(seasons)
    d = d[(d.play_deleted != 1) | d.play_deleted.isna()]
    d["is_scrimmage"] = (
        d.play_type.isin(["pass", "run"])
        & (d.qb_kneel != 1)
        & (d.qb_spike != 1)
        & (d.two_point_attempt != 1)
        & d.down.notna()
        & d.epa.notna()
        & d.posteam.notna()
        & (d.qtr <= 4)
    ).astype(int)
    d["success"] = (d.epa > 0).astype(int)
    d["half"] = np.where(d.qtr <= 2, 1, 2)
    d["dome"] = d.roof.isin(["dome", "closed"]).astype(int)

    # Pre-play scoreboard (total_*_score is after the play).
    for side in ["home", "away"]:
        col = f"total_{side}_score"
        d[f"pre_{side}"] = d.groupby("game_id")[col].shift(1).fillna(0)

    d = add_pregame_strength(d)

    # ---------- plays ----------
    p = d[d.is_scrimmage == 1].copy()
    p["first_play_of_drive"] = (
        p.groupby(["game_id", "fixed_drive"]).cumcount() == 0
    ).astype(int)
    p["lev"] = p.wpa.abs()
    keep = """game_id season season_type week home_team away_team posteam defteam play_id fixed_drive qtr half
    down ydstogo yardline_100 goal_to_go game_seconds_remaining half_seconds_remaining score_differential
    posteam_timeouts_remaining defteam_timeouts_remaining pass epa success wp wpa lev home pos_spread
    pos_implied def_implied total_line dome temp wind off_sr_pre def_sr_pre first_play_of_drive
    interception sack fumble_lost fourth_down_converted passer_player_id passer_player_name desc""".split()
    p = p[keep].reset_index(drop=True)
    p["seq"] = p.groupby(["game_id", "posteam"]).cumcount()
    p.to_parquet(CLEAN / "nfl_plays.parquet", index=False)

    # ---------- drives ----------
    d["drive_key"] = d.game_id + "_" + d.fixed_drive.astype("Int64").astype(str)
    first = d.groupby("drive_key").head(1).set_index("drive_key")
    # next drive's first row gives the score after this drive (incl. PAT/2pt)
    drive_order = first.reset_index()[["drive_key", "game_id", "fixed_drive"]].sort_values(["game_id", "fixed_drive"])
    drive_order["next_key"] = drive_order.groupby("game_id").drive_key.shift(-1)

    final = d.groupby("game_id").agg(fh=("total_home_score", "last"), fa=("total_away_score", "last"))
    start_h = first.pre_home
    start_a = first.pre_away
    drive_order["start_h"] = drive_order.drive_key.map(start_h)
    drive_order["start_a"] = drive_order.drive_key.map(start_a)
    drive_order["end_h"] = drive_order.next_key.map(start_h)
    drive_order["end_a"] = drive_order.next_key.map(start_a)
    miss = drive_order.end_h.isna()
    drive_order.loc[miss, "end_h"] = drive_order.loc[miss, "game_id"].map(final.fh)
    drive_order.loc[miss, "end_a"] = drive_order.loc[miss, "game_id"].map(final.fa)

    scr = d[d.is_scrimmage == 1]
    agg = scr.groupby("drive_key").agg(
        game_id=("game_id", "first"), season=("season", "first"), season_type=("season_type", "first"),
        week=("week", "first"), posteam=("posteam", "first"), defteam=("defteam", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        fixed_drive=("fixed_drive", "first"), qtr=("qtr", "first"), half=("half", "first"),
        start_yardline_100=("yardline_100", "first"), start_half_secs=("half_seconds_remaining", "first"),
        start_game_secs=("game_seconds_remaining", "first"), start_score_diff=("score_differential", "first"),
        pos_to=("posteam_timeouts_remaining", "first"), def_to=("defteam_timeouts_remaining", "first"),
        end_qtr=("qtr", "last"), end_half_secs=("half_seconds_remaining", "last"),
        n_plays=("play_id", "size"), sr=("success", "mean"), epa_sum=("epa", "sum"),
        home=("home", "first"), pos_spread=("pos_spread", "first"), pos_implied=("pos_implied", "first"),
        def_implied=("def_implied", "first"), total_line=("total_line", "first"), dome=("dome", "first"),
        off_sr_pre=("off_sr_pre", "first"), def_sr_pre=("def_sr_pre", "first"),
        start_wp=("wp", "first"),
    ).reset_index()
    res = d.groupby("drive_key").fixed_drive_result.first()
    agg["result"] = agg.drive_key.map(res)
    agg = agg.merge(drive_order[["drive_key", "start_h", "start_a", "end_h", "end_a"]], on="drive_key", how="left")
    home = agg.posteam == agg.home_team
    pf = np.where(home, agg.end_h - agg.start_h, agg.end_a - agg.start_a)
    pa = np.where(home, agg.end_a - agg.start_a, agg.end_h - agg.start_h)
    agg["points"] = pf - pa
    # Seconds the drive took (within a half; drives that cross a half are rare and dropped below).
    agg["duration"] = agg.start_half_secs - agg.end_half_secs
    agg = agg[(agg.qtr <= 4) & (agg.half == np.where(agg.end_qtr <= 2, 1, 2))]
    agg = agg.sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    agg.to_parquet(CLEAN / "nfl_drives.parquet", index=False)

    # ---------- fumbles ----------
    f = d[
        (d.fumble == 1) & d.play_type.isin(["pass", "run"]) & (d.aborted_play != 1)
        & (d.fumble_out_of_bounds != 1) & (d.qtr <= 4) & d.posteam.notna() & (d.two_point_attempt != 1)
    ].copy()
    f["lost"] = (f.fumble_lost == 1).astype(int)
    f["ftype"] = np.select([f.sack == 1, f.play_type == "pass", f.play_type == "run"], ["sack", "reception", "rush"], "other")
    f["drive_key"] = f.game_id + "_" + f.fixed_drive.astype("Int64").astype(str)
    f = f[["game_id", "season", "season_type", "week", "posteam", "defteam", "home_team", "play_id", "order_sequence",
           "fixed_drive", "drive_key", "qtr", "half", "down", "ydstogo", "yardline_100", "half_seconds_remaining",
           "game_seconds_remaining", "score_differential", "wp", "home", "pos_spread", "pos_implied", "def_implied",
           "off_sr_pre", "def_sr_pre", "fumble_forced", "lost", "ftype", "return_touchdown", "desc"]]
    f.to_parquet(CLEAN / "nfl_fumbles.parquet", index=False)
    print(f"plays {len(p):,}  drives {len(agg):,}  fumbles {len(f):,}")


if __name__ == "__main__":
    build()
