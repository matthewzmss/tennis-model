#!/usr/bin/env python3
"""
ATP Tennis Scorecard Model (learned from data)
==============================================

A points-based scorecard for ATP matches whose weights are LEARNED from
historical results (logistic regression), not set by hand.

Every factor gives "points" to whichever player has the edge. 10 points = 1 unit
of log-odds, so the total converts straight into a win probability and
American odds.

Data: Jeff Sackmann's tennis_atp (https://github.com/JeffSackmann/tennis_atp).
Put the extracted GitHub ZIP in ./tennis_data (any subfolder is fine).
NOTE: that data is licensed CC BY-NC-SA 4.0 (non-commercial).

Defaults: 2006-2010 warm up ratings only; weights are learned from 2011 onward,
with recent seasons weighted more (half-life 4 years; --half-life 0 = off).

Data sources
------------
* Sackmann / Tennis Abstract (GitHub ZIP in ./tennis_data): detailed serve and
  return stats, style types, rally proxy, court speed. Lags weeks-months.
* tennis-data.co.uk yearly files: download 2025.xlsx and 2026.xlsx by hand from
  tennis-data.co.uk/data.php into ./tennis_data (re-download weekly). Their terms
  say private use only, no automated bots - so this script never downloads them.
  Recent results with exact dates -> Elo, form, streak, fatigue, rest, H2H.
* recent_results.csv (optional, yours): anything still missing. Columns:
  date,tournament,surface,round,winner,loser,score   e.g.
  2026-10-01,Tokyo,Hard,F,Casper Ruud,Some Player,6-4 6-3
* predict checks every player's last match date and neutralizes form/fatigue
  if it's stale, so a hot streak the data hasn't seen is never read as bad form.

Rules the model follows
-----------------------
* Every feature uses ONLY matches played before the match being predicted.
* Testing is walk-forward: train on years < Y, test on year Y, roll forward.
* The model is compared with plain Elo.
* Sportsbook odds are never used.

Style: opponents are typed from their stats at match time - big server (ace
rate), big returner (return pts won), grinder / short-point player (top / bottom
25% in seconds per point among active players, a rally-length proxy) - and each player's record vs.
each type is a factor. Court speed is computed from ace rates vs. expectation
at each event in previous years (1.0 = average).

Factors the box score can't measure (forehand/backhand quality, exact rally
lengths, net play, injuries, motivation, coaching...) go in as MANUAL point
adjustments at predict time: --adjust-a 3 (player A +3 points, ~+7% near 50/50)

Usage
-----
  pip install pandas numpy scikit-learn openpyxl
  python tennis_model.py train
  python tennis_model.py predict "Jannik Sinner" "Carlos Alcaraz" --surface Hard
  python tennis_model.py predict "Sinner" "Zverev" --surface Clay --best-of 5 \
        --tourney "Madrid Masters" --adjust-b -4 --note "Zverev back issue"
"""
import argparse
import json
import math
import os
import pickle
import socket
import sys
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

socket.setdefaulttimeout(20)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
MIRRORS = [
    BASE_URL,
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main/",
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main/atp/",
]
DATA_DIR = "tennis_data"
OUT_DIR = "tennis_output"

TOUR_LEVELS = {"G", "M", "A", "F"}
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 2, "R128": 3, "R64": 4,
               "R32": 5, "R16": 6, "QF": 7, "SF": 8, "BR": 8, "F": 9,
               "RR": 6, "ER": 3}
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# ----------------------------------------------------------------------------
# Tournament lookups (matched by substring of the tournament name, lowercase).
# Partial lists covering the main ATP events; unknown events get no effect.
# ----------------------------------------------------------------------------
TOURNEY_COUNTRY = {
    "australian open": "AUS", "brisbane": "AUS", "adelaide": "AUS", "sydney": "AUS",
    "roland garros": "FRA", "paris": "FRA", "marseille": "FRA", "montpellier": "FRA",
    "metz": "FRA", "lyon": "FRA", "nice": "FRA",
    "wimbledon": "GBR", "queen": "GBR", "eastbourne": "GBR", "nottingham": "GBR",
    "us open": "USA", "indian wells": "USA", "miami": "USA", "cincinnati": "USA",
    "washington": "USA", "atlanta": "USA", "winston": "USA", "houston": "USA",
    "delray": "USA", "dallas": "USA", "memphis": "USA", "san jose": "USA",
    "newport": "USA", "new york": "USA", "san diego": "USA",
    "los cabos": "MEX", "acapulco": "MEX",
    "madrid": "ESP", "barcelona": "ESP", "mallorca": "ESP", "valencia": "ESP", "gijon": "ESP",
    "rome": "ITA", "florence": "ITA", "naples": "ITA", "sardinia": "ITA", "cagliari": "ITA",
    "parma": "ITA",
    "hamburg": "GER", "halle": "GER", "munich": "GER", "stuttgart": "GER",
    "vienna": "AUT", "kitzbuhel": "AUT",
    "gstaad": "SUI", "basel": "SUI", "geneva": "SUI",
    "rotterdam": "NED", "hertogenbosch": "NED", "antwerp": "BEL",
    "stockholm": "SWE", "bastad": "SWE", "umag": "CRO", "zagreb": "CRO",
    "doha": "QAT", "dubai": "UAE", "beijing": "CHN", "shanghai": "CHN",
    "chengdu": "CHN", "zhuhai": "CHN", "tokyo": "JPN", "canada": "CAN",
    "toronto": "CAN", "montreal": "CAN", "buenos aires": "ARG", "cordoba": "ARG",
    "rio de janeiro": "BRA", "sao paulo": "BRA", "santiago": "CHI", "bogota": "COL",
    "quito": "ECU", "estoril": "POR", "sofia": "BUL", "belgrade": "SRB",
    "budapest": "HUN", "auckland": "NZL", "pune": "IND", "chennai": "IND",
    "marrakech": "MAR", "casablanca": "MAR", "astana": "KAZ", "almaty": "KAZ",
    "st. petersburg": "RUS", "moscow": "RUS", "kremlin": "RUS", "istanbul": "TUR",
    "seoul": "KOR", "hong kong": "HKG", "banja luka": "BIH", "tel aviv": "ISR",
}
TOURNEY_ALTITUDE_M = {
    "bogota": 2600, "quito": 2850, "gstaad": 1050, "kitzbuhel": 760, "sao paulo": 760,
    "madrid": 650, "santiago": 570, "munich": 520, "cordoba": 400, "geneva": 375,
}
TOURNEY_INDOOR = [
    "paris masters", "tour finals", "next gen", "basel", "vienna", "rotterdam",
    "marseille", "montpellier", "sofia", "stockholm", "antwerp", "metz", "dallas",
    "memphis", "st. petersburg", "moscow", "kremlin", "zagreb", "astana",
]


def tourney_lookup(name):
    n = str(name).lower()
    country = next((v for k, v in sorted(TOURNEY_COUNTRY.items(), key=lambda kv: -len(kv[0]))
                    if k in n), None)
    alt = next((v for k, v in TOURNEY_ALTITUDE_M.items() if k in n), 0)
    indoor = any(k in n for k in TOURNEY_INDOOR)
    return country, alt, indoor


# Priors (tour averages) used to shrink small samples
PRIOR = {"first_in": 0.62, "first_won": 0.73, "second_won": 0.52,
         "ret_first": 0.27, "ret_second": 0.48, "hold": 0.80, "brk": 0.20,
         "ace": 0.075, "df": 0.035}
SHRINK_PTS = 300
SHRINK_GMS = 40
SURF_SHRINK = 400
BIG_SERVER_ACE = 0.10      # ace rate that makes an opponent a "big server"
BIG_RETURNER_RET = 0.40    # return pts won that makes an opponent a "big returner"
STYLE_PCT = 25            # top/bottom 25% in seconds per point = grinder / short-point player
FAST_COURT, SLOW_COURT = 1.10, 0.90   # court speed ratings (1.0 = tour average)

FEATURES = [
    # Ratings / opponent quality
    ("elo",              "Overall Elo (per 100 pts)"),
    ("surface_elo",      "Surface Elo (per 100 pts)"),
    ("margin_elo",       "Margin Elo - rewards blowouts (per 100)"),
    ("rank",             "Ranking, log ratio"),
    ("opp_quality",      "Avg opponent Elo, last 12 mo (per 100)"),
    # Recent performance / confidence
    ("form",             "Win rate, last 10 matches"),
    ("form_vs_exp",      "Results vs. Elo expectation, last 15"),
    ("streak",           "Current win/loss streak"),
    # Serve & return (surface-adjusted, last 12 months)
    ("first_in",         "1st serve in %"),
    ("first_won",        "1st serve points won %"),
    ("second_won",       "2nd serve points won %"),
    ("ret_first",        "Return pts won vs 1st serve %"),
    ("ret_second",       "Return pts won vs 2nd serve %"),
    ("hold",             "Hold %"),
    ("break",            "Break %"),
    ("aces",             "Ace rate (per serve point, %)"),
    ("double_faults",    "Double fault rate (%)"),
    # Pressure / format
    ("tiebreak",         "Tiebreak win rate (last 24 mo)"),
    ("deciding_set",     "Deciding-set win rate (last 24 mo)"),
    ("bo5_stamina",      "Best-of-5 record vs. expectation (Slams)"),
    ("elo_bo5",          "Elo edge x best-of-5"),
    # Serve-vs-return matchup (each serve battle scored head-on)
    ("ace_clash",        "Free points: expected ace rate in THIS matchup"),
    ("rally_clash",      "Rally battle: win % once the serve comes back"),
    ("serve_matchup",    "Serve vs returners like this opponent (vs expected)"),
    ("return_matchup",   "Return vs servers like this opponent (vs expected)"),
    ("points_model",     "Point-by-point model: match win chance (log-odds)"),
    ("rally_length",     "Avg seconds per point (rally-length proxy)"),
    ("lefty",            "Lefty vs righty edge"),
    ("vs_lefty",         "Record vs lefties (when facing one)"),
    # Head to head
    ("h2h",              "Head-to-head wins (shrunk)"),
    ("h2h_margin",       "H2H games-won margin"),
    # Physical / schedule
    ("fatigue_hours",    "Court hours, last 7 days"),
    ("matches_14d",      "Matches played, last 14 days"),
    ("rest_days",        "Days since last match (capped 60)"),
    ("prev_match_min",   "Previous round match length (hours)"),
    ("tourney_hours",    "Court hours so far this tournament"),
    ("tourney_sets",     "Sets played so far this tournament"),
    ("retirements",      "Retired mid-match, last 12 mo (injury proxy)"),
    ("age",              "Age (years)"),
    ("height",           "Height (cm)"),
    ("experience",       "Matches last 12 mo (log)"),
    ("surface_matches",  "Matches on this surface this season"),
    # Tournament / court
    ("tourney_history",  "Past results at this tournament vs. expectation"),
    ("court_speed_pref", "Fast/slow-court record x this court's speed"),
    ("serve_x_speed",    "Serve edge x this court's speed"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
# Factors that are meaningless if a player's recent matches are missing
RECENCY_FEATURES = ["form", "form_vs_exp", "streak", "fatigue_hours", "matches_14d",
                    "rest_days", "prev_match_min", "tourney_hours", "tourney_sets",
                    "surface_matches"]
FEATURE_DESC = dict(FEATURES)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
_index = None


def find_local(fname, data_dir):
    global _index
    if _index is None:
        _index = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith((".csv", ".xlsx", ".xls")):
                    _index.setdefault(f, os.path.join(root, f))
    return _index.get(fname)


def download(url, path, refresh=False):
    """Download to a temp name first. An existing file is only replaced when
    refresh=True AND the new download succeeds; it is never deleted."""
    if os.path.exists(path) and not refresh:
        return True
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < 1000:          # error page, not data
            raise ValueError("download too small")
        os.replace(tmp, path)
        return True
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return os.path.exists(path)


def fetch(fname, data_dir, missing):
    local = find_local(fname, data_dir) if os.path.isdir(data_dir) else None
    if local:
        return local
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)
    for base in MIRRORS:
        if download(base + fname, path):
            print(f"  downloaded {fname}")
            return path
    missing.append(fname)
    return None


def load_matches(data_dir, start, end, challengers):
    frames, missing = [], []
    for y in range(start, end + 1):
        names = [(f"atp_matches_{y}.csv", "tour")]
        if challengers:
            names.append((f"atp_matches_qual_chall_{y}.csv", "chall"))
        for fname, src in names:
            path = fetch(fname, data_dir, missing)
            if path is None:
                continue
            df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
            df["source"] = src
            frames.append(df)
    tour_found = sum(1 for f in frames if len(f) and f["source"].iloc[0] == "tour")
    if missing:
        print(f"  {len(missing)} files not found (e.g. {missing[0]}); "
              f"{tour_found} tour-level years loaded")
    if not frames or tour_found == 0:
        sys.exit(
            "\nNo match data found.\n"
            "Fix: open https://github.com/JeffSackmann/tennis_atp in your browser,\n"
            "click the green 'Code' button > 'Download ZIP', and extract it into\n"
            f"the '{data_dir}' folder next to this script (any subfolder is fine).")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["winner_id"].notna() & df["loser_id"].notna()]
    df["score"] = df["score"].fillna("").astype(str)
    df = df[~df["score"].str.contains("W/O|walkover|Walkover", regex=True)]
    df["date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    df["round_idx"] = df["round"].map(ROUND_ORDER).fillna(5)
    span = np.where(df["tourney_level"] == "G", 1.4, 0.8)
    df["day"] = df["date"] + pd.to_timedelta(df["round_idx"] * span, unit="D")
    df["surface"] = df["surface"].fillna("Hard")
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    df["tourney_name"] = df["tourney_name"].fillna("").astype(str)
    df = df.sort_values(["day", "tourney_id", "round_idx", "match_num"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Current results: tennis-data.co.uk + your own recent_results.csv
# (Sackmann data lags; these keep Elo, form, streak, fatigue and H2H current.)
# ----------------------------------------------------------------------------
TD_ROUND = {"1st Round": "R128", "2nd Round": "R64", "3rd Round": "R32", "4th Round": "R16",
            "Quarterfinals": "QF", "Semifinals": "SF", "The Final": "F", "Round Robin": "RR"}
TD_LEVEL = {"Grand Slam": "G", "Masters 1000": "M", "Masters": "M", "Masters Cup": "F",
            "ATP500": "A", "ATP250": "A", "International Gold": "A", "International": "A"}


def norm(txt):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().lower()
    return " ".join(t.replace("-", " ").replace(".", " ").replace("'", " ").split())


class NameMap:
    """Matches 'Ruud C.' (tennis-data) or 'Casper Ruud' to Sackmann player IDs."""
    def __init__(self, df):
        self.full, self.short, self.latest = {}, defaultdict(set), {}
        for side in ("winner", "loser"):
            sub = df[[f"{side}_id", f"{side}_name", "day"]].dropna()
            for pid, name, day in sub.itertuples(index=False):
                self.full[norm(name)] = pid
                toks = norm(name).split()
                if len(toks) >= 2:
                    for k in range(1, len(toks)):
                        self.short[(" ".join(toks[k:]), toks[0][0])].add(pid)
                if day > self.latest.get(pid, pd.Timestamp(0)):
                    self.latest[pid] = day
        self.new_ids = {}

    def get(self, name):
        n = norm(name)
        if n in self.full:
            return self.full[n]
        toks = n.split()
        cands = set()
        if len(toks) >= 2 and len(toks[-1]) <= 2:            # "ruud c" / "de minaur a"
            initials = toks[-1]
            cands = self.short.get((" ".join(toks[:-1]), initials[0]), set())
        if len(cands) == 1:
            return next(iter(cands))
        if len(cands) > 1:                                   # pick most recently active
            return max(cands, key=lambda p: self.latest.get(p, pd.Timestamp(0)))
        if n not in self.new_ids:                            # unknown player: new ID
            self.new_ids[n] = f"new:{n}"
        return self.new_ids[n]


def pick_up_from_downloads(y, data_dir):
    """If you downloaded e.g. '2026.xlsx' or '2026 (3).xlsx' in your browser, copy the
    newest one from your Downloads folder into data_dir. No moving files by hand."""
    import glob
    import shutil
    dl = os.path.join(os.path.expanduser("~"), "Downloads")
    import re
    pat = re.compile(rf"^{y}(\D.*)?\.(xlsx|xls)$", re.IGNORECASE)   # 2026.xlsx, 2026 (2).xlsx, 2026-2.xlsx
    cands = [f for f in glob.glob(os.path.join(dl, "*"))
             if pat.match(os.path.basename(f)) and not os.path.basename(f).endswith(".crdownload")]
    if not cands:
        return
    newest = max(cands, key=os.path.getmtime)
    os.makedirs(data_dir, exist_ok=True)
    dest = find_local(f"{y}.xlsx", data_dir) or os.path.join(data_dir, f"{y}.xlsx")
    if not os.path.exists(dest) or os.path.getmtime(newest) > os.path.getmtime(dest):
        shutil.copy2(newest, dest)
        if _index is not None:
            _index[f"{y}.xlsx"] = dest
        print(f"  picked up {os.path.basename(newest)} from Downloads")


def load_td_year(y, data_dir):
    """Reads a tennis-data.co.uk file you downloaded in your browser (Downloads folder
    or data_dir). Their terms ask that files not be fetched by automated bots, so
    this script never downloads them itself."""
    pick_up_from_downloads(y, data_dir)
    for fname in (f"{y}.xlsx", f"{y}.xls", f"{y}.csv"):
        path = find_local(fname, data_dir) if os.path.isdir(data_dir) else None
        if path:
            break
    else:
        if y >= datetime.now().year:
            print(f"  tennis-data {y}.xlsx not found - click {y} at tennis-data.co.uk/data.php"
                  " (it will be picked up from your Downloads folder)")
        return None
    days = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
    if y >= datetime.now().year:
        print(f"  tennis-data {y}: using {os.path.basename(path)} (downloaded {days:.0f} days ago)"
              + ("  <- older than a week: click 2026 on tennis-data.co.uk/data.php again" if days > 7 else ""))
    try:
        return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    except Exception as e:
        print(f"  could not read {path}: {e} (pip install openpyxl)")
        return None


def td_to_rows(td, names):
    rows = []
    for r in td.to_dict("records"):
        comment = str(r.get("Comment", "Completed"))
        if "Walkover" in comment or pd.isna(r.get("Winner")) or pd.isna(r.get("Loser")):
            continue
        sets = []
        for i in range(1, 6):
            wg, lg = num(r.get(f"W{i}")), num(r.get(f"L{i}"))
            if wg is not None and lg is not None:
                sets.append(f"{int(wg)}-{int(lg)}")
        score = " ".join(sets) + (" RET" if "Retired" in comment else "")
        day = pd.Timestamp(r["Date"])
        tname = str(r.get("Tournament", r.get("Location", "")))
        rows.append({
            "tourney_id": f"td-{day.year}-{norm(tname)}", "tourney_name": tname,
            "surface": str(r.get("Surface", "Hard")), "tourney_level": TD_LEVEL.get(str(r.get("Series")), "A"),
            "tourney_date": int(day.strftime("%Y%m%d")), "match_num": 0,
            "winner_id": names.get(r["Winner"]), "winner_name": str(r["Winner"]),
            "loser_id": names.get(r["Loser"]), "loser_name": str(r["Loser"]),
            "winner_rank": num(r.get("WRank")), "loser_rank": num(r.get("LRank")),
            "score": score, "best_of": int(num(r.get("Best of")) or 3),
            "round": TD_ROUND.get(str(r.get("Round")), "R32"), "day": day, "source": "tour",
            "data_source": "tennis-data",
        })
    return rows


def manual_to_rows(path, names):
    m = pd.read_csv(path)
    rows = []
    for r in m.to_dict("records"):
        day = pd.Timestamp(r["date"])
        tname = str(r.get("tournament", "Manual"))
        rows.append({
            "tourney_id": f"man-{day.year}-{norm(tname)}", "tourney_name": tname,
            "surface": str(r.get("surface", "Hard")), "tourney_level": str(r.get("level", "A")),
            "tourney_date": int(day.strftime("%Y%m%d")), "match_num": 0,
            "winner_id": names.get(r["winner"]), "winner_name": str(r["winner"]),
            "loser_id": names.get(r["loser"]), "loser_name": str(r["loser"]),
            "score": str(r.get("score", "")), "best_of": int(num(r.get("best_of")) or 3),
            "round": str(r.get("round", "R32")), "minutes": num(r.get("minutes")),
            "day": day, "source": "tour", "data_source": "manual",
        })
    return rows


def merge_current(df, data_dir, years, manual_path):
    """Add matches that the Sackmann data doesn't have yet (deduplicated)."""
    names = NameMap(df)
    extra = []
    for y in years:
        td = load_td_year(y, data_dir)
        if td is not None:
            extra += td_to_rows(td, names)
    if manual_path and os.path.exists(manual_path):
        extra += manual_to_rows(manual_path, names)
        print(f"  read {manual_path}")
    if not extra:
        print("  no current-results files found (tennis-data.co.uk / recent_results.csv)")
        return df
    ex = pd.DataFrame(extra)
    # Dedupe: same winner & loser within 21 days of a match we already have
    have = defaultdict(list)
    for w, l, d in zip(df["winner_id"], df["loser_id"], df["day"]):
        have[(w, l)].append(d)
    keep = []
    for w, l, d in zip(ex["winner_id"], ex["loser_id"], ex["day"]):
        keep.append(not any(abs((d - x).days) <= 21 for x in have.get((w, l), [])))
    ex = ex[keep].drop_duplicates(subset=["winner_id", "loser_id", "day"])
    if len(ex) == 0:
        print("  current-results files add no new matches")
        return df
    ex["round_idx"] = ex["round"].map(ROUND_ORDER).fillna(5)
    ex["date"] = ex["day"]
    new_players = sum(1 for v in names.new_ids)
    print(f"  added {len(ex):,} newer matches from current-results files "
          f"(through {ex['day'].max().date()}); {new_players} players not in Sackmann data")
    df = df.assign(data_source="sackmann")
    out = pd.concat([df, ex], ignore_index=True)
    return out.sort_values(["day", "tourney_id", "round_idx", "match_num"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Score parsing
# ----------------------------------------------------------------------------
def parse_score(score, best_of):
    """Returns dict: winner/loser tiebreaks, games, deciding set, retired."""
    retired = any(t in score for t in ("RET", "ABD", "DEF", "Def"))
    wtb = ltb = sets = wg = lg = 0
    for tok in score.split():
        if "-" not in tok or tok.startswith("["):
            continue
        a, b = tok.split("-", 1)
        b = b.split("(")[0]
        try:
            a, b = int(a), int(b.strip("]"))
        except ValueError:
            continue
        sets += 1
        wg += a; lg += b
        if (a, b) == (7, 6):
            wtb += 1
        elif (a, b) == (6, 7):
            ltb += 1
    return {"wtb": wtb, "ltb": ltb, "wg": wg, "lg": lg, "sets": sets,
            "deciding": (sets == best_of) and not retired, "retired": retired}


# ----------------------------------------------------------------------------
# Player state
# ----------------------------------------------------------------------------
def k_factor(n):
    return 250.0 / ((n + 5) ** 0.4)


def elo_prob(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


STAT_KEYS = ("svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpFaced", "bpSaved", "ace", "df")

# ---------------------------------------------------------------------------
# Point-by-point (Markov) model: serve-point win chances -> game -> set -> match
# ---------------------------------------------------------------------------
def p_game(p):
    q = 1 - p
    deuce = p * p / (p * p + q * q)
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


def p_tiebreak(pa, pb):
    """A serves first point; serve alternates every two points after that."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def f(a, b):
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a >= 6 and a == b:                      # from 6-6: pairs of points
            w, l = pa * (1 - pb), (1 - pa) * pb
            return w / (w + l) if w + l > 0 else 0.5
        n = a + b
        a_serves = (n % 4 == 0) or (n % 4 == 3)
        p = pa if a_serves else 1 - pb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)


def p_set(pa, pb):
    """pa/pb = chance each player wins a point on his own serve. A serves first."""
    from functools import lru_cache
    ga, gb = p_game(pa), 1 - p_game(pb)            # A holds / A breaks
    tb = p_tiebreak(pa, pb)

    @lru_cache(maxsize=None)
    def f(a, b):
        if a >= 6 and a - b >= 2 or a == 7:
            return 1.0
        if b >= 6 and b - a >= 2 or b == 7:
            return 0.0
        if a == 6 and b == 6:
            return tb
        g = ga if (a + b) % 2 == 0 else gb
        return g * f(a + 1, b) + (1 - g) * f(a, b + 1)
    return f(0, 0)


def p_match(pa, pb, best_of=3):
    s = 0.5 * (p_set(pa, pb) + (1 - p_set(pb, pa)))  # average over who serves first
    need = best_of // 2 + 1
    from math import comb
    return sum(comb(need - 1 + k, k) * s**need * (1 - s)**k for k in range(need))


class State:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.n = defaultdict(int)
        self.selo = defaultdict(lambda: 1500.0)
        self.sn = defaultdict(int)
        self.melo = defaultdict(lambda: 1500.0)      # margin-of-victory Elo
        self.hist = defaultdict(deque)               # pid -> matches (last 2 yrs)
        self.h2h = defaultdict(int)                  # (a,b) -> wins of a over b
        self.h2h_games = defaultdict(int)            # (a,b) -> games a won vs b
        self.tourney_res = defaultdict(lambda: [0.0, 0])   # (pid, tourney) -> [sum resid, n]
        self.ace_ema = defaultdict(lambda: PRIOR["ace"])
        self.ret_ema = defaultdict(lambda: 0.37)
        self.ace_allowed_ema = defaultdict(lambda: PRIOR["ace"])
        # Rally ability: points won when the serve is NOT an ace
        self.svna_ema = defaultdict(lambda: 0.60)   # on own serve, once returned/played
        self.rtna_ema = defaultdict(lambda: 0.40)   # on return, of non-ace points
        self.g_ace, self.g_svna = PRIOR["ace"], 0.60  # tour averages (running)
        # Same profile kept per surface (blended with overall by sample size)
        self.prof_s = {}                              # (pid, surface) -> dict
        self.g_s = {}                                 # surface -> tour averages
        self.h2h_s = defaultdict(int)                 # (a, b, surface) -> wins
        self.global_spp = 40.0                       # tour-average seconds per point
        self.rally_ema = {}                          # pid -> seconds per point
        self.rally_cut = (1e9, -1e9)                 # (short <=, grinder >=) thresholds
        self._updates = 0
        self.speed = defaultdict(lambda: [0.0, 0.0])        # (tourney, year) -> [aces, expected]
        self.surf_speed = defaultdict(lambda: [0.0, 0.0])   # surface -> [aces, expected]
        self.info = {}
        self.last_rank = {}
        self.last_day = None

    def blended(self, pid, surface):
        return 0.5 * self.elo[pid] + 0.5 * self.selo[(pid, surface)]

    def blended_elo_prob(self, a, b, surface):
        return elo_prob(self.blended(a, surface), self.blended(b, surface))

    # ---- serve-vs-return matchup -----------------------------------------
    SURF_PROF_K = 6       # surface matches needed before the surface profile counts half

    def profile(self, pid, surface):
        """Style profile on this surface, blended with the overall profile when the
        player has few matches on it. Returns (ace, aces_allowed, svna, rtna, n)."""
        overall = (self.ace_ema[pid], self.ace_allowed_ema[pid],
                   self.svna_ema[pid], self.rtna_ema[pid])
        ps = self.prof_s.get((pid, surface))
        if not ps:
            return overall + (0,)
        w = ps["n"] / (ps["n"] + self.SURF_PROF_K)
        vals = tuple(w * ps[k] + (1 - w) * o for k, o in zip(("ace", "aa", "svna", "rtna"), overall))
        return vals + (ps["n"],)

    def surface_avg(self, surface):
        return self.g_s.get(surface, {"ace": self.g_ace, "svna": self.g_svna})

    def serve_profile(self, pid, surface):
        pr = self.profile(pid, surface)
        return (pr[0], pr[2])

    def return_profile(self, pid, surface):
        pr = self.profile(pid, surface)
        return (pr[1], pr[3])

    def speed_mult(self, cs, surface, year):
        """This court's speed relative to an average court of the same surface."""
        base = self.court_speed("__surface_average__", year, surface)
        return min(1.6, max(0.6, cs / base)) if base > 0 else 1.0

    def predict_serve(self, srv, ret, surface, mult=1.0):
        """Chance `srv` wins a point on serve vs `ret` on this surface/court, split
        into the ace battle and the rally battle (points where the serve isn't an ace)."""
        ps, pr = self.profile(srv, surface), self.profile(ret, surface)
        g = self.surface_avg(surface)
        ace = min(0.40, ps[0] * pr[1] / max(g["ace"], 1e-3) * mult)
        rally = ps[2] - (pr[3] - (1 - g["svna"]))
        rally = min(0.85, max(0.30, rally))
        return ace + (1 - ace) * rally, ace, rally

    def is_big_server(self, pid):
        return self.ace_ema[pid] >= BIG_SERVER_ACE

    def is_big_returner(self, pid):
        return self.ret_ema[pid] >= BIG_RETURNER_RET

    def is_grinder(self, pid):
        return pid in self.rally_ema and self.rally_ema[pid] >= self.rally_cut[1]

    def is_short_points(self, pid):
        return pid in self.rally_ema and self.rally_ema[pid] <= self.rally_cut[0]

    def _refresh_rally_cuts(self, day):
        # percentiles among players active in the last year (relative to each other)
        recent = [self.rally_ema[p] for p in self.rally_ema
                  if self.hist[p] and self.hist[p][-1]["day"] >= day - timedelta(days=365)]
        if len(recent) >= 40:
            self.rally_cut = (float(np.percentile(recent, STYLE_PCT)),
                              float(np.percentile(recent, 100 - STYLE_PCT)))

    def court_speed(self, tkey, year, surface):
        """Ace rate vs. what these servers/returners usually produce, from the
        event's previous 3 years, shrunk toward the surface average."""
        sa, se = self.surf_speed[surface]
        surf = (sa + 50 * 1.0) / (se + 50)
        act = exp = 0.0
        for y in range(year - 3, year):
            a_, e_ = self.speed.get((tkey, y), (0.0, 0.0))
            act += a_; exp += e_
        return (act + 150 * surf) / (exp + 150)

    def matchup_residual(self, pid, day, kind, target, surface=None):
        """How `pid` did vs opponents SIMILAR to `target` profile, relative to what
        the serve/return model expected. kind='serve' compares past opponents'
        return profiles; kind='return' compares their serve profiles."""
        key_p, key_r = ("opp_ret_prof", "sv_res") if kind == "serve" else ("opp_srv_prof", "rt_res")
        sw = swr = 0.0
        for m in self.hist[pid]:
            if m["day"] >= day or key_p not in m:
                continue
            p0, p1 = m[key_p]
            d2 = ((p0 - target[0]) / 0.025) ** 2 + ((p1 - target[1]) / 0.03) ** 2
            w = math.exp(-d2 / 2) * (2.0 if m["surface"] == surface else 1.0)
            sw += w; swr += w * m[key_r]
        return swr / (sw + 4.0), sw          # shrunk toward 0; sw = effective matches

    # ---- per-player summary at a point in time ---------------------------
    def player_stats(self, pid, day, surface, tourney_id, tourney_key):
        h = self.hist[pid]
        cutoff = day - timedelta(days=730)
        while h and h[0]["day"] < cutoff:
            h.popleft()
        y1 = day - timedelta(days=365)
        agg, sagg = defaultdict(float), defaultdict(float)
        n1 = m14 = surf_season = rets = 0
        mins7 = 0.0
        tbw = tbl = dw = dl = 0
        opp_elo_sum = 0.0
        bo5 = [0.0, 0]; vl = [0.0, 0]; vs = [0.0, 0]; vr = [0.0, 0]
        vg = [0.0, 0]; vsp = [0.0, 0]; fast = [0.0, 0]; slow = [0.0, 0]
        sec = pts = ssec = spts = 0.0
        t_min = 0.0; t_sets = 0
        past = [m for m in h if m["day"] < day]
        for m in past:
            tbw += m["tbw"]; tbl += m["tbl"]; dw += m["dw"]; dl += m["dl"]
            resid = m["won"] - m["exp"]
            if m["bo"] == 5:
                bo5[0] += resid; bo5[1] += 1
            if m["opp_lefty"]:
                vl[0] += resid; vl[1] += 1
            if m["opp_big_server"]:
                vs[0] += resid; vs[1] += 1
            if m["opp_big_returner"]:
                vr[0] += resid; vr[1] += 1
            if m["opp_grinder"]:
                vg[0] += resid; vg[1] += 1
            if m["opp_short"]:
                vsp[0] += resid; vsp[1] += 1
            if m["court_speed"] >= FAST_COURT:
                fast[0] += resid; fast[1] += 1
            elif m["court_speed"] <= SLOW_COURT:
                slow[0] += resid; slow[1] += 1
            if m["tourney"] == tourney_id:
                t_min += m["minutes"]; t_sets += m["sets"]
            if m["day"] >= day - timedelta(days=7):
                mins7 += m["minutes"]
            if m["day"] >= day - timedelta(days=14):
                m14 += 1
            if m["surface"] == surface and m["day"].year == day.year:
                surf_season += 1
            if m["day"] < y1:
                continue
            n1 += 1
            opp_elo_sum += m["opp_elo"]
            if m["spp_pts"] > 0:
                sec += m["spp_sec"]; pts += m["spp_pts"]
                if m["surface"] == surface:
                    ssec += m["spp_sec"]; spts += m["spp_pts"]
            rets += m["ret_loss"]
            if m["has_stats"]:
                for k, v in m["st"].items():
                    agg[k] += v
                    if m["surface"] == surface:
                        sagg[k] += v

        def r(num_, den, prior, k):
            return (num_ + k * prior) / (den + k)

        def both(numk, denk, prior, k=SHRINK_PTS, invert=False):
            def val(a):
                nm = (a[denk] - a[numk]) if invert else a[numk]
                return nm, a[denk]
            n_all, d_all = val(agg)
            overall = r(n_all, d_all, prior, k)
            n_s, d_s = val(sagg)
            return r(n_s, d_s, overall, SURF_SHRINK if k == SHRINK_PTS else k)

        # derived denominators
        for a in (agg, sagg):
            a["2nd_pts"] = a["svpt"] - a["1stIn"]
            a["o_2nd_pts"] = a["o_svpt"] - a["o_1stIn"]
            a["sv_held"] = a["SvGms"] - (a["bpFaced"] - a["bpSaved"])
            a["o_broken"] = a["o_bpFaced"] - a["o_bpSaved"]

        s = {
            "first_in": both("1stIn", "svpt", PRIOR["first_in"]),
            "first_won": both("1stWon", "1stIn", PRIOR["first_won"]),
            "second_won": both("2ndWon", "2nd_pts", PRIOR["second_won"]),
            "ret_first": both("o_1stWon", "o_1stIn", PRIOR["ret_first"], invert=True),
            "ret_second": both("o_2ndWon", "o_2nd_pts", PRIOR["ret_second"], invert=True),
            "hold": both("sv_held", "SvGms", PRIOR["hold"], k=SHRINK_GMS),
            "brk": both("o_broken", "o_SvGms", PRIOR["brk"], k=SHRINK_GMS),
            "ace": both("ace", "svpt", PRIOR["ace"]),
            "df": both("df", "svpt", PRIOR["df"]),
        }
        s["serve_pts"] = s["first_in"] * s["first_won"] + (1 - s["first_in"]) * s["second_won"]

        last10 = [m["won"] for m in past][-10:]
        last15 = past[-15:]
        streak = 0
        for m in reversed(past):
            if streak == 0:
                streak = 1 if m["won"] else -1
            elif (streak > 0) == bool(m["won"]):
                streak += 1 if streak > 0 else -1
            else:
                break
        prev = past[-1] if past else None
        tr = self.tourney_res[(pid, tourney_key)]
        s.update({
            "form": (sum(last10) + 2.5) / (len(last10) + 5),
            "form_vs_exp": sum(m["won"] - m["exp"] for m in last15) / (len(last15) + 5),
            "streak": max(-8, min(8, streak)),
            "mins7": mins7 / 60.0, "m14": m14,
            "rest": 60.0 if prev is None else min(60.0, (day - prev["day"]).days),
            "prev_min": (prev["minutes"] / 60.0) if (prev and prev["tourney"] == tourney_id) else 0.0,
            "tb": (tbw + 3) / (tbw + tbl + 6),
            "dec": (dw + 3) / (dw + dl + 6),
            "bo5": bo5[0] / (bo5[1] + 10),
            "vs_lefty": vl[0] / (vl[1] + 8),
            "vs_server": vs[0] / (vs[1] + 8),
            "vs_returner": vr[0] / (vr[1] + 8),
            "opp_quality": ((opp_elo_sum + 5 * 1500) / (n1 + 5) - 1500) / 100.0,
            "exp": math.log1p(n1),
            "surf_season": surf_season,
            "retirements": rets,
            "tourney_hist": tr[0] / (tr[1] + 5),
            "vs_grinder": vg[0] / (vg[1] + 8),
            "vs_short": vsp[0] / (vsp[1] + 8),
            "speed_pref": fast[0] / (fast[1] + 8) - slow[0] / (slow[1] + 8),
            "spp": (ssec + 800 * ((sec + 1500 * self.global_spp) / (pts + 1500))) / (spts + 800),
            "t_hours": t_min / 60.0, "t_sets": t_sets,
        })
        return s

    def features(self, a, b, day, surface, best_of, tourney_id, tourney_name,
                 info_a, info_b, rank_a, rank_b):
        tkey = str(tourney_name).lower()
        sa = self.player_stats(a, day, surface, tourney_id, tkey)
        sb = self.player_stats(b, day, surface, tourney_id, tkey)
        cs = self.court_speed(tkey, day.year, surface)
        elo_d = (self.elo[a] - self.elo[b]) / 100.0
        ra = rank_a if rank_a else 1500.0
        rb = rank_b if rank_b else 1500.0
        age_a, age_b = info_a.get("age"), info_b.get("age")
        ht_a, ht_b = info_a.get("ht"), info_b.get("ht")
        la, lb = info_a.get("hand") == "L", info_b.get("hand") == "L"
        # H2H: meetings on this surface count double
        wa = self.h2h[(a, b)] + self.h2h_s[(a, b, surface)]
        wb = self.h2h[(b, a)] + self.h2h_s[(b, a, surface)]
        ga, gb = self.h2h_games[(a, b)], self.h2h_games[(b, a)]
        serve_edge = (sa["serve_pts"] - sb["serve_pts"]) * 100
        bs_a, bs_b = self.is_big_server(a), self.is_big_server(b)
        br_a, br_b = self.is_big_returner(a), self.is_big_returner(b)
        # Serve-vs-return battles
        mult = self.speed_mult(cs, surface, day.year)
        spw_ab, ace_ab, rally_ab = self.predict_serve(a, b, surface, mult)
        spw_ba, ace_ba, rally_ba = self.predict_serve(b, a, surface, mult)
        g_svna = self.surface_avg(surface)["svna"]
        sv_m_a, _ = self.matchup_residual(a, day, "serve", self.return_profile(b, surface), surface)
        sv_m_b, _ = self.matchup_residual(b, day, "serve", self.return_profile(a, surface), surface)
        rt_m_a, _ = self.matchup_residual(a, day, "return", self.serve_profile(b, surface), surface)
        rt_m_b, _ = self.matchup_residual(b, day, "return", self.serve_profile(a, surface), surface)
        pa = min(0.95, max(0.30, spw_ab + sv_m_a - rt_m_b))
        pb = min(0.95, max(0.30, spw_ba + sv_m_b - rt_m_a))
        pm = min(0.999, max(0.001, p_match(pa, pb, best_of)))
        self._last_matchup = {"spw_ab": spw_ab, "ace_ab": ace_ab, "rally_ab": rally_ab,
                              "spw_ba": spw_ba, "ace_ba": ace_ba, "rally_ba": rally_ba,
                              "sv_m_a": sv_m_a, "sv_m_b": sv_m_b, "rt_m_a": rt_m_a,
                              "rt_m_b": rt_m_b, "pa": pa, "pb": pb, "pm": pm,
                              "mult": mult, "cs": cs}
        d = lambda k, mult=100: (sa[k] - sb[k]) * mult
        return [
            elo_d,
            (self.selo[(a, surface)] - self.selo[(b, surface)]) / 100.0,
            (self.melo[a] - self.melo[b]) / 100.0,
            math.log(rb / ra),
            d("opp_quality", 1),
            d("form"),
            d("form_vs_exp"),
            d("streak", 1),
            d("first_in"), d("first_won"), d("second_won"),
            d("ret_first"), d("ret_second"),
            d("hold"), d("brk"), d("ace"), d("df"),
            d("tb"), d("dec"),
            d("bo5") * (1.0 if best_of == 5 else 0.0),
            elo_d * (1.0 if best_of == 5 else 0.0),
            (ace_ab - ace_ba) * 100,
            ((1 - ace_ab) * (rally_ab - g_svna) - (1 - ace_ba) * (rally_ba - g_svna)) * 100,
            (sv_m_a - sv_m_b) * 100,
            (rt_m_a - rt_m_b) * 100,
            math.log(pm / (1 - pm)),
            d("spp", 1),
            float(la and not lb) - float(lb and not la),
            (sa["vs_lefty"] * lb - sb["vs_lefty"] * la) * 100,
            (wa - wb) / (wa + wb + 2),
            (ga - gb) / (ga + gb + 20),
            d("mins7", 1), d("m14", 1), d("rest", 1), d("prev_min", 1),
            d("t_hours", 1), d("t_sets", 1),
            d("retirements", 1),
            (age_a - age_b) if (age_a and age_b) else 0.0,
            (ht_a - ht_b) if (ht_a and ht_b) else 0.0,
            d("exp", 1),
            d("surf_season", 1),
            d("tourney_hist"),
            d("speed_pref") * (cs - 1.0) * 10,
            serve_edge * (cs - 1.0),
        ]

    # ---- update after a match ---------------------------------------------
    def update(self, row, sc):
        w, l, s = row["winner_id"], row["loser_id"], row["surface"]
        tkey = str(row["tourney_name"]).lower()
        exp_w = self.blended_elo_prob(w, l, s)
        pre = {"elo_w": self.blended(w, s), "elo_l": self.blended(l, s),
               "lefty_w": self.info.get(w, {}).get("hand") == "L",
               "lefty_l": self.info.get(l, {}).get("hand") == "L",
               "bs_w": self.is_big_server(w), "bs_l": self.is_big_server(l),
               "br_w": self.is_big_returner(w), "br_l": self.is_big_returner(l),
               "gr_w": self.is_grinder(w), "gr_l": self.is_grinder(l),
               "sh_w": self.is_short_points(w), "sh_l": self.is_short_points(l)}
        cs = self.court_speed(tkey, row["day"].year, s)
        mult = self.speed_mult(cs, s, row["day"].year)
        pre["spw_w"] = self.predict_serve(w, l, s, mult)[0]
        pre["spw_l"] = self.predict_serve(l, w, s, mult)[0]
        pre["rp_w"], pre["rp_l"] = self.return_profile(w, s), self.return_profile(l, s)
        pre["sp_w"], pre["sp_l"] = self.serve_profile(w, s), self.serve_profile(l, s)
        # Elo
        p = elo_prob(self.elo[w], self.elo[l])
        self.elo[w] += k_factor(self.n[w]) * (1 - p)
        self.elo[l] -= k_factor(self.n[l]) * (1 - p)
        ps = elo_prob(self.selo[(w, s)], self.selo[(l, s)])
        self.selo[(w, s)] += k_factor(self.sn[(w, s)]) * (1 - ps)
        self.selo[(l, s)] -= k_factor(self.sn[(l, s)]) * (1 - ps)
        # Margin Elo: actual score from game share (blowout ~1, tight ~0.55)
        tot = sc["wg"] + sc["lg"]
        g = sc["wg"] / tot if tot > 0 and not sc["retired"] else 0.55
        actual = min(1.0, max(0.0, 0.5 + 2.5 * (g - 0.5)))
        pm = elo_prob(self.melo[w], self.melo[l])
        self.melo[w] += k_factor(self.n[w]) * (actual - pm)
        self.melo[l] -= k_factor(self.n[l]) * (actual - pm)
        self.n[w] += 1; self.n[l] += 1
        self.sn[(w, s)] += 1; self.sn[(l, s)] += 1
        # H2H
        self.h2h[(w, l)] += 1
        self.h2h_s[(w, l, s)] += 1
        self.h2h_games[(w, l)] += sc["wg"]; self.h2h_games[(l, w)] += sc["lg"]
        # Tournament history (residual vs expectation)
        for pid, won, e in ((w, 1, exp_w), (l, 0, 1 - exp_w)):
            tr = self.tourney_res[(pid, tkey)]
            tr[0] += won - e; tr[1] += 1

        stats = {f"{side}_{k}": num(row.get(f"{side}_{k}")) for side in ("w", "l") for k in STAT_KEYS}
        has = all(v is not None for v in stats.values()) and (stats["w_svpt"] or 0) > 0 \
            and (stats["l_svpt"] or 0) > 0 and (stats["w_SvGms"] or 0) > 0 and (stats["l_SvGms"] or 0) > 0
        minutes = num(row.get("minutes")) or 0.0
        tot_pts = (stats["w_svpt"] or 0) + (stats["l_svpt"] or 0) if has else 0
        spp = minutes * 60 / tot_pts if (has and minutes > 20 and tot_pts > 40) else 0.0
        if 15 < spp < 90:
            self.global_spp = 0.999 * self.global_spp + 0.001 * spp
        else:
            spp = 0.0
        if has:   # court speed: actual aces vs. expected for these server/returner pairs
            sp = self.speed[(tkey, row["day"].year)]; ss = self.surf_speed[s]
            for srv, ret, side in ((w, l, "w"), (l, w, "l")):
                e = stats[f"{side}_svpt"] * self.ace_ema[srv] * self.ace_allowed_ema[ret] / PRIOR["ace"]
                sp[0] += stats[f"{side}_ace"]; sp[1] += e
                ss[0] += stats[f"{side}_ace"]; ss[1] += e
        for me, opp, won, my, op in ((w, l, 1, "w", "l"), (l, w, 0, "l", "w")):
            rec = {"day": row["day"], "surface": s, "won": won, "minutes": minutes,
                   "tourney": row["tourney_id"], "bo": row["best_of"],
                   "exp": exp_w if won else 1 - exp_w,
                   "opp_elo": pre[f"elo_{op}"],
                   "opp_lefty": pre[f"lefty_{op}"],
                   "opp_big_server": pre[f"bs_{op}"],
                   "opp_big_returner": pre[f"br_{op}"],
                   "opp_grinder": pre[f"gr_{op}"], "opp_short": pre[f"sh_{op}"],
                   "court_speed": cs, "sets": sc["sets"],
                   "spp_sec": spp * tot_pts if spp else 0.0, "spp_pts": tot_pts if spp else 0,
                   "tbw": sc["wtb"] if won else sc["ltb"], "tbl": sc["ltb"] if won else sc["wtb"],
                   "dw": int(sc["deciding"] and won), "dl": int(sc["deciding"] and not won),
                   "ret_loss": int(sc["retired"] and not won),
                   "has_stats": has}
            if has:
                st = {k: stats[f"{my}_{k}"] for k in STAT_KEYS}
                st.update({f"o_{k}": stats[f"{op}_{k}"] for k in STAT_KEYS})
                rec["st"] = st
                # serve-vs-return residuals vs. what the matchup model expected
                my_spw = (st["1stWon"] + st["2ndWon"]) / st["svpt"]
                opp_spw = (st["o_1stWon"] + st["o_2ndWon"]) / st["o_svpt"]
                rec["opp_ret_prof"] = pre[f"rp_{op}"]
                rec["opp_srv_prof"] = pre[f"sp_{op}"]
                rec["sv_res"] = my_spw - pre[f"spw_{my}"]
                rec["rt_res"] = (1 - opp_spw) - (1 - pre[f"spw_{op}"])
                # rally ability: points won when the serve wasn't an ace
                na_sv = st["svpt"] - st["ace"]
                na_ret = st["o_svpt"] - st["o_ace"]
                if na_sv > 0:
                    v = (st["1stWon"] + st["2ndWon"] - st["ace"]) / na_sv
                    self.svna_ema[me] = 0.85 * self.svna_ema[me] + 0.15 * v
                    self.g_svna = 0.9995 * self.g_svna + 0.0005 * v
                if na_ret > 0:
                    v = (st["o_svpt"] - st["o_1stWon"] - st["o_2ndWon"]) / na_ret
                    self.rtna_ema[me] = 0.85 * self.rtna_ema[me] + 0.15 * v
                self.g_ace = 0.9995 * self.g_ace + 0.0005 * (st["ace"] / st["svpt"])
                # style profiles (EMA) for future opponents' "vs big server" stats
                self.ace_ema[me] = 0.85 * self.ace_ema[me] + 0.15 * (st["ace"] / st["svpt"])
                ret = 1 - (st["o_1stWon"] + st["o_2ndWon"]) / st["o_svpt"]
                self.ret_ema[me] = 0.85 * self.ret_ema[me] + 0.15 * ret
                self.ace_allowed_ema[me] = (0.85 * self.ace_allowed_ema[me]
                                            + 0.15 * (st["o_ace"] / st["o_svpt"]))
                # same profile for this surface
                ps = self.prof_s.get((me, s))
                if ps is None:
                    ps = {"ace": st["ace"] / st["svpt"], "aa": st["o_ace"] / st["o_svpt"],
                          "svna": self.svna_ema[me], "rtna": self.rtna_ema[me], "n": 0}
                    self.prof_s[(me, s)] = ps
                new = {"ace": st["ace"] / st["svpt"], "aa": st["o_ace"] / st["o_svpt"]}
                if na_sv > 0:
                    new["svna"] = (st["1stWon"] + st["2ndWon"] - st["ace"]) / na_sv
                if na_ret > 0:
                    new["rtna"] = (st["o_svpt"] - st["o_1stWon"] - st["o_2ndWon"]) / na_ret
                for k_, v_ in new.items():
                    ps[k_] = 0.8 * ps[k_] + 0.2 * v_
                ps["n"] += 1
                g = self.g_s.setdefault(s, {"ace": self.g_ace, "svna": self.g_svna})
                g["ace"] = 0.999 * g["ace"] + 0.001 * new["ace"]
                if "svna" in new:
                    g["svna"] = 0.999 * g["svna"] + 0.001 * new["svna"]
            if spp:
                prev = self.rally_ema.get(me, self.global_spp)
                self.rally_ema[me] = 0.85 * prev + 0.15 * spp
            self.hist[me].append(rec)
        self._updates += 1
        if self._updates % 1000 == 0:
            self._refresh_rally_cuts(row["day"])


def player_info(row, side):
    return {"name": row.get(f"{side}_name"), "hand": row.get(f"{side}_hand"),
            "ht": num(row.get(f"{side}_ht")), "age": num(row.get(f"{side}_age")),
            "ioc": row.get(f"{side}_ioc")}


# ----------------------------------------------------------------------------
# Build the feature table by replaying history in order
# ----------------------------------------------------------------------------
def build_dataset(df, eval_start, seed=7):
    rng = np.random.default_rng(seed)
    st = State()
    rows = []
    for row in df.to_dict("records"):
        sc = parse_score(row["score"], row["best_of"])
        w, l = row["winner_id"], row["loser_id"]
        iw, il = player_info(row, "winner"), player_info(row, "loser")
        for pid, inf in ((w, iw), (l, il)):       # static info; never overwrite with blanks
            old = st.info.get(pid, {})
            merged = dict(old)
            for k, v in inf.items():
                blank = v is None or (isinstance(v, float) and math.isnan(v))
                if not blank and not (k == "name" and old.get("name")
                                      and row.get("data_source") in ("tennis-data", "manual")):
                    merged[k] = v
            if inf.get("age") is not None:
                merged["age_day"] = row["day"]
            st.info[pid] = merged

        def current(pid):                          # known info, age advanced to today
            i = dict(st.info[pid])
            if i.get("age") and i.get("age_day") is not None:
                i["age"] += (row["day"] - i["age_day"]).days / 365.25
            return i
        iw, il = current(w), current(l)
        rw, rl = num(row.get("winner_rank")), num(row.get("loser_rank"))
        eligible = (row["source"] == "tour" and row["tourney_level"] in TOUR_LEVELS
                    and row["day"].year >= eval_start and not sc["retired"])
        if eligible:
            if rng.random() < 0.5:
                a, b, ia, ib, ra, rb, y = w, l, iw, il, rw, rl, 1
            else:
                a, b, ia, ib, ra, rb, y = l, w, il, iw, rl, rw, 0
            x = st.features(a, b, row["day"], row["surface"], row["best_of"],
                            row["tourney_id"], row["tourney_name"], ia, ib, ra, rb)
            pe = st.blended_elo_prob(a, b, row["surface"])
            rows.append([row["day"].year, row["day"], ia["name"], ib["name"],
                         row["surface"], row["tourney_name"], row["round"], y, pe] + x)
        st.update(row, sc)                         # AFTER features: no leakage
        if rw: st.last_rank[w] = rw
        if rl: st.last_rank[l] = rl
        st.last_day = row["day"]
    cols = ["year", "day", "player_a", "player_b", "surface", "tourney", "round",
            "a_won", "elo_prob"] + FEATURE_NAMES
    return pd.DataFrame(rows, columns=cols), st


# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
HALF_LIFE = 4.0


def recency_weights(years):
    if years is None or not HALF_LIFE:
        return None
    years = np.asarray(years, dtype=float)
    return 0.5 ** ((years.max() - years) / HALF_LIFE)


def fit_lr(X, y, C, years=None):
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    m.fit(X, y, sample_weight=recency_weights(years))
    return m


def scale_fit(X):
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return sd


def choose_C(data, train_years, folds=3):
    """Pick regularization by averaging log loss over the last `folds` training
    years (each validated with a model trained on the years before it)."""
    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    val_years = sorted(train_years)[-folds:]
    scores = defaultdict(list)
    for vy in val_years:
        tr = data[data.year.isin([y for y in train_years if y < vy])]
        va = data[data.year == vy]
        if len(tr) < 500 or len(va) < 100:
            continue
        sd = scale_fit(tr[FEATURE_NAMES].values)
        for C in grid:
            m = fit_lr(tr[FEATURE_NAMES].values / sd, tr.a_won.values, C, tr.year.values)
            p = m.predict_proba(va[FEATURE_NAMES].values / sd)[:, 1]
            scores[C].append(log_loss(va.a_won.values, p, labels=[0, 1]))
    if not scores:
        return 0.1
    return min(scores, key=lambda C: np.mean(scores[C]))


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(y), "accuracy": accuracy_score(y, p > 0.5),
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p)}


def walk_forward(data, train_start, test_years):
    out = []
    for Y in test_years:
        train_years = list(range(train_start, Y))
        tr = data[data.year.isin(train_years)]
        te = data[data.year == Y].copy()
        if len(te) == 0 or len(tr) < 500:
            continue
        C = choose_C(data, train_years)
        sd = scale_fit(tr[FEATURE_NAMES].values)
        m = fit_lr(tr[FEATURE_NAMES].values / sd, tr.a_won.values, C, tr.year.values)
        te["model_prob"] = m.predict_proba(te[FEATURE_NAMES].values / sd)[:, 1]
        e = ["elo", "surface_elo"]
        sde = scale_fit(tr[e].values)
        me = fit_lr(tr[e].values / sde, tr.a_won.values, 1.0, tr.year.values)
        te["elo_lr_prob"] = me.predict_proba(te[e].values / sde)[:, 1]
        te["C"] = C
        out.append(te)
        print(f"  {Y}: {len(te)} matches  (C={C})")
    return pd.concat(out) if out else pd.DataFrame()


def american(p):
    p = min(max(p, 0.001), 0.999)
    return f"-{round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_train(args):
    global HALF_LIFE
    HALF_LIFE = args.half_life
    os.makedirs(args.out, exist_ok=True)
    end = args.end or datetime.now().year
    print(f"Loading ATP data {args.start}-{end} ...")
    df = load_matches(args.data, args.start, end, not args.no_challengers)
    print(f"  {len(df):,} matches loaded (Sackmann, through {df['day'].max().date()})")
    if not args.no_current:
        print("Adding current results (tennis-data.co.uk, recent_results.csv) ...")
        df = merge_current(df, args.data, range(max(args.start, end - 2), end + 1), args.recent)
    print("Replaying history and computing pre-match features ...")
    data, st = build_dataset(df, args.train_start)
    print(f"  {len(data):,} tour-level matches with features")

    last_full = int(data.year.max())
    test_years = list(range(max(args.train_start + 3, last_full - args.test_years + 1), last_full + 1))
    print(f"Walk-forward testing on {test_years[0]}-{test_years[-1]} ...")
    bt = walk_forward(data, args.train_start, test_years)
    bt.to_csv(os.path.join(args.out, "backtest.csv"), index=False)

    lines = ["ATP TENNIS MODEL - WALK-FORWARD BACKTEST",
             f"Training from {args.train_start}, recency half-life: "
             + (f"{HALF_LIFE:g} years" if HALF_LIFE else "off")
             + f", {len(FEATURE_NAMES)} factors",
             "(lower log loss / Brier = better; trained only on earlier years)", ""]

    def block(label, d):
        r = []
        for name, col in (("Scorecard model", "model_prob"), ("Elo-only (fitted)", "elo_lr_prob"),
                          ("Raw blended Elo", "elo_prob")):
            mm = metrics(d.a_won.values, d[col].values)
            r.append(f"  {name:<20} acc {mm['accuracy']:.3f}   log loss {mm['log_loss']:.4f}   brier {mm['brier']:.4f}")
        return [f"{label} (n={len(d)})"] + r + [""]
    lines += block("ALL TEST YEARS", bt)
    for Y in test_years:
        d = bt[bt.year == Y]
        if len(d):
            lines += block(str(Y), d)
    for s in SURFACES:
        d = bt[bt.surface == s]
        if len(d) > 100:
            lines += block(f"Surface: {s}", d)
    lines.append("CALIBRATION (model prob bucket -> actual win rate)")
    bt["bucket"] = (bt.model_prob.clip(0, 0.999) * 10).astype(int) / 10
    for bkt, g in bt.groupby("bucket"):
        lines.append(f"  {bkt:.1f}-{bkt + 0.1:.1f}: predicted {g.model_prob.mean():.3f}  actual {g.a_won.mean():.3f}  (n={len(g)})")

    all_years = list(range(args.train_start, last_full + 1))
    C = choose_C(data, all_years)
    sd = scale_fit(data[FEATURE_NAMES].values)
    m = fit_lr(data[FEATURE_NAMES].values / sd, data.a_won.values, C, data.year.values)
    coef = m.coef_[0]
    w = pd.DataFrame({
        "factor": FEATURE_NAMES,
        "description": [FEATURE_DESC[f] for f in FEATURE_NAMES],
        "weight_per_std": coef,
        "points_per_unit": coef / sd * 10,
        "typical_spread": sd,
        "importance": np.abs(coef),
    }).sort_values("importance", ascending=False)
    w.to_csv(os.path.join(args.out, "weights.csv"), index=False)
    lines += ["", f"FINAL MODEL (C={C}) - LEARNED WEIGHTS, most important first",
              "  weight_per_std: effect of a typical-size edge (compare factors with this)",
              "  points_per_unit: scorecard points per 1 unit of edge",
              "  (10 points = 1 log-odds; +10 total ~ 73%, +20 ~ 88%)",
              "  Negative weights on overlapping stats usually mean 'already counted",
              "  elsewhere', not 'worse is better'.", ""]
    for _, r in w.iterrows():
        lines.append(f"  {r.description:<46} {r.weight_per_std:+.3f}/std   {r.points_per_unit:+.3f} pts per unit")
    report = "\n".join(lines)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out, "model.json"), "w") as f:
        json.dump({"features": FEATURE_NAMES, "coef": coef.tolist(), "scale": sd.tolist(),
                   "C": C, "half_life": HALF_LIFE,
                   "trained_through": str(data.day.max().date())}, f, indent=2)
    with open(os.path.join(args.out, "state.pkl"), "wb") as f:
        pickle.dump({k: (dict(v) if isinstance(v, defaultdict) else v)
                     for k, v in st.__dict__.items()}, f)
    print("\n" + report)
    print(f"\nSaved to ./{args.out}/")


def load_state(out):
    with open(os.path.join(out, "state.pkl"), "rb") as f:
        d = pickle.load(f)
    st = State()
    for k, v in d.items():
        cur = getattr(st, k, None)
        if isinstance(cur, defaultdict):
            cur.update(v)
        else:
            setattr(st, k, v)
    return st


def find_player(st, name):
    q = name.lower()
    exact = [pid for pid, i in st.info.items() if str(i.get("name", "")).lower() == q]
    if exact:
        return exact[0]
    hits = [pid for pid, i in st.info.items() if q in str(i.get("name", "")).lower()]
    if not hits:
        sys.exit(f"Player not found: {name}")
    if len(hits) > 1:
        hits.sort(key=lambda p: -st.n.get(p, 0))
        print(f"  '{name}' matched {len(hits)} players; using {st.info[hits[0]]['name']}")
    return hits[0]


def cmd_predict(args):
    with open(os.path.join(args.out, "model.json")) as f:
        model = json.load(f)
    if model["features"] != FEATURE_NAMES:
        sys.exit("Model was trained with a different version of this script. Run 'train' again.")
    st = load_state(args.out)
    a, b = find_player(st, args.player_a), find_player(st, args.player_b)
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp(datetime.now().date())
    ia, ib = dict(st.info[a]), dict(st.info[b])
    for i in (ia, ib):
        if i.get("age") and i.get("age_day") is not None:
            i["age"] += max(0.0, (day - pd.Timestamp(i["age_day"])).days / 365.25)
    x = np.array(st.features(a, b, day, args.surface, args.best_of, "upcoming", args.tourney,
                             ia, ib, st.last_rank.get(a), st.last_rank.get(b)))
    # Freshness guard. Two different situations:
    #  1) the DATA is old (files not updated)  -> recent-form factors are unknown -> neutral
    #  2) data is current but a PLAYER hasn't played -> real layoff -> keep layoff factors
    data_through = pd.Timestamp(st.last_day)
    data_age = (day - data_through).days
    data_stale = data_age > args.stale_days
    print("\nDATA FRESHNESS")
    print(f"  Results data current through {data_through.date()}"
          + (f"  WARNING: {data_age} days old - re-download 2026.xlsx from tennis-data.co.uk"
             if data_stale else "  OK"))
    layoffs = []
    for pid, info in ((a, ia), (b, ib)):
        last = st.hist[pid][-1]["day"] if st.hist[pid] else None
        gap = (day - last).days if last is not None else 999
        where = st.hist[pid][-1]["tourney"] if st.hist[pid] else "-"
        status = "OK"
        if gap > args.layoff_days and not data_stale:
            status = f"LAYOFF: no matches in {gap} days (injury or break?)"
            layoffs.append(info["name"])
        elif gap > args.stale_days:
            status = f"unknown - data is old ({gap} days since last match in data)"
        print(f"  {info['name']:<24} last match: {last.date() if last is not None else 'none'} "
              f"({where})  {status}")
    if data_stale:
        for f_ in RECENCY_FEATURES:
            x[FEATURE_NAMES.index(f_)] = 0.0
        print("  -> Form, streak, fatigue and rest set to NEUTRAL because the data itself is old.")
    if layoffs:
        print(f"  -> {', '.join(layoffs)}: the layoff is counted (rest days), but the model can't")
        print("     see WHY. Check the news; for an injury return, consider --adjust-a/--adjust-b.")
    # Style profile + the two serve battles
    na_, nb_ = ia["name"], ib["name"]
    print(f"\nSTYLE PROFILE on {args.surface} (recent matches, blended with all-surface"
          " numbers when a player has few matches on it)")
    print(f"  {'':<24}{'Aces':>7}{'Rally won':>11}{'Aces vs':>9}{'Rally won':>11}{'Surface':>9}")
    print(f"  {'':<24}{'(serve)':>7}{'on serve':>11}{'(return)':>9}{'on return':>11}{'matches':>9}")
    for pid, info in ((a, ia), (b, ib)):
        pr = st.profile(pid, args.surface)
        print(f"  {info['name']:<24}{pr[0]:>7.1%}{pr[2]:>11.1%}{pr[1]:>9.1%}{pr[3]:>11.1%}{pr[4]:>9}")
    g_ = st.surface_avg(args.surface)
    print(f"  {'tour average (' + args.surface + ')':<24}{g_['ace']:>7.1%}{g_['svna']:>11.1%}"
          f"{g_['ace']:>9.1%}{1 - g_['svna']:>11.1%}")
    print("  (Rally won = points won when the serve is NOT an ace; Aces vs = aces")
    print("   conceded on return, lower = better returner)")
    m = st._last_matchup
    print(f"\nSERVE BATTLES  (court speed {m['cs']:.2f}; aces x{m['mult']:.2f} vs an average"
          f" {args.surface.lower()} court)")
    for srv, ret, k, sv_m, rt_m, final in ((na_, nb_, "ab", m["sv_m_a"], m["rt_m_b"], m["pa"]),
                                           (nb_, na_, "ba", m["sv_m_b"], m["rt_m_a"], m["pb"])):
        print(f"  On {srv}'s serve vs {ret}:")
        print(f"    free points: {m['ace_' + k]:.1%} aces expected"
              f" | rallies: {srv} wins {m['rally_' + k]:.1%} once it comes back")
        print(f"    -> {srv} wins {m['spw_' + k]:.1%} of serve points (stats model)")
        print(f"    history: {srv}'s serve vs returners like {ret}: {sv_m * 100:+.1f} pts vs expected;"
              f" {ret}'s return vs servers like {srv}: {rt_m * 100:+.1f}")
        print(f"    -> used in point-by-point model: {final:.1%}")
    print(f"  Point-by-point model: {na_} {m['pm']:.1%} to win the match")
    coef, sd = np.array(model["coef"]), np.array(model["scale"])
    pts = coef * (x / sd) * 10
    model_total = pts.sum()
    manual = args.adjust_a - args.adjust_b
    total = model_total + manual
    p = 1 / (1 + math.exp(-total / 10))
    na, nb = ia["name"], ib["name"]
    print(f"\n{na} vs {nb}  |  {args.surface}, best of {args.best_of}"
          + (f"  |  {args.tourney}" if args.tourney else "") + f"  |  as of {day.date()}")
    print(f"Model trained through {model['trained_through']}\n")
    print(f"{'Factor':<46}{'Edge':>10}{'Points':>9}  Favors")
    for name, val, pt in sorted(zip(FEATURE_NAMES, x, pts), key=lambda t: -abs(t[2])):
        who = na if pt > 0.05 else (nb if pt < -0.05 else "even")
        print(f"{FEATURE_DESC[name]:<46}{val:>10.2f}{pt:>+9.2f}  {who}")
    print(f"\nModel scorecard: {abs(model_total):.1f} points toward {na if model_total >= 0 else nb}")
    if manual:
        print(f"Manual adjustment: {na} {args.adjust_a:+g}, {nb} {args.adjust_b:+g}"
              + (f"  ({args.note})" if args.note else ""))
        print(f"Final scorecard: {abs(total):.1f} points toward {na if total >= 0 else nb}")
    print(f"Win probability: {na} {p:.1%} ({american(p)})  |  {nb} {1 - p:.1%} ({american(1 - p)})")
    pe = st.blended_elo_prob(a, b, args.surface)
    print(f"Plain Elo for comparison: {na} {pe:.1%}")


def main():
    ap = argparse.ArgumentParser(description="ATP tennis scorecard model")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--start", type=int, default=2006, help="first year loaded (warm-up, not trained on)")
    t.add_argument("--train-start", type=int, default=2011, help="first year used for training")
    t.add_argument("--end", type=int, default=None)
    t.add_argument("--test-years", type=int, default=6)
    t.add_argument("--half-life", type=float, default=4.0,
                   help="recency weighting in years: a match this many years older counts half (0 = off)")
    t.add_argument("--no-challengers", action="store_true")
    t.add_argument("--no-current", action="store_true", help="skip tennis-data.co.uk / recent_results.csv")
    t.add_argument("--recent", default="recent_results.csv",
                   help="your own file of recent matches: date,tournament,surface,round,winner,loser,score")
    p = sub.add_parser("predict")
    p.add_argument("player_a"); p.add_argument("player_b")
    p.add_argument("--surface", default="Hard", choices=SURFACES)
    p.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    p.add_argument("--tourney", default="", help='e.g. "Madrid Masters" (court speed, tournament history)')
    p.add_argument("--date", default=None, help="match date YYYY-MM-DD (default: today)")
    p.add_argument("--stale-days", type=int, default=10,
                   help="results data older than this = out of date (form/fatigue neutralized)")
    p.add_argument("--layoff-days", type=int, default=21,
                   help="a player with no matches for this long (data current) = layoff")
    p.add_argument("--adjust-a", type=float, default=0.0,
                   help="manual points for player A (injury, motivation, conditions...)")
    p.add_argument("--adjust-b", type=float, default=0.0, help="manual points for player B")
    p.add_argument("--note", default="", help="reason for the manual adjustment")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    main()
