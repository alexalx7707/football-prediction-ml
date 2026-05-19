"""
Router for match prediction endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import os
import logging
import pandas as pd
import numpy as np
import joblib
import kagglehub

logger = logging.getLogger("football-api.prediction")

router = APIRouter(prefix="/predict", tags=["Predictions"])


# ──────────────────────────────────────────────
# 📊 PYDANTIC MODELS
# ──────────────────────────────────────────────

class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team")
    away_team: str = Field(..., description="Away team")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Manchester United",
                "away_team": "Liverpool"
            }
        }


class ReasoningFactor(BaseModel):
    id: str = Field(..., description="Short factor identifier (elo, form_last5, h2h, ...)")
    label: str = Field(..., description="Human-readable label for the factor")
    home_value: float = Field(..., description="Factor value for the home team")
    away_value: float = Field(..., description="Factor value for the away team")
    favors: str = Field(..., description="Which side this factor favors: 'home', 'away', or 'none'")
    description: str = Field(..., description="Short description of the factor")


class PredictionReasoning(BaseModel):
    summary: str = Field(..., description="Narrative explanation of the model's pick")
    factors: List[ReasoningFactor] = Field(..., description="Structured factors for the UI to render")


class PredictionResponse(BaseModel):
    match: str
    prediction: str
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    elo_difference: float
    probabilities: Dict[str, float]
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    reasoning: PredictionReasoning


# ──────────────────────────────────────────────
# 🔧 HELPER FUNCTIONS
# ──────────────────────────────────────────────

def load_models():
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        le = joblib.load('models/label_encoder.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        try:
            meta_learner = joblib.load('models/meta_learner.pkl')
        except Exception:
            meta_learner = None
        try:
            draw_model = joblib.load('models/draw_specialist_model.pkl')
        except Exception:
            draw_model = None
        return rf_model, lr_model, xgb_model, le, feature_columns, meta_learner, draw_model
    except Exception:
        return None, None, None, None, None, None, None


def load_elo_baseline():
    """Mean league Elo baseline saved during training; falls back to 1500 if missing."""
    try:
        val = float(joblib.load('models/league_elo_baseline.pkl'))
        if np.isfinite(val) and val > 0:
            return val
    except Exception:
        pass
    return 1500.0


def get_team_elo(team_name, elo_df=None, matches_df=None, is_home=True):
    if elo_df is not None:
        team_elo = elo_df[elo_df['club'] == team_name].sort_values('date')
        if len(team_elo) > 0:
            return float(team_elo['elo'].iloc[-1])
    if matches_df is not None:
        col = 'HomeTeam' if is_home else 'AwayTeam'
        elo_col = 'HomeElo' if is_home else 'AwayElo'
        team_matches = matches_df[matches_df[col] == team_name].sort_values('MatchDate')
        if len(team_matches) > 0:
            return float(team_matches[elo_col].iloc[-1])
    return 1500.0


ROLLING_STATS = [
    'GoalsScored', 'GoalsConceded', 'Points', 'GoalDiff', 'Won', 'ShotsOnTarget', 'IsDraw', 'XG',
    # Opponent-Elo-weighted variants — values scaled by OppElo/league_baseline at the per-match level
    'GoalsScored_EloWtd', 'GoalsConceded_EloWtd', 'Points_EloWtd', 'GoalDiff_EloWtd', 'XG_EloWtd',
]
LEGACY_STATS = ['Shots', 'Target', 'Corners', 'Fouls', 'Yellow', 'Red', 'Form3', 'Form5',
                'GoalsScored', 'GoalsConceded', 'Points', 'GoalDiff', 'XG']


def _team_long_history(team_name, matches_df, elo_baseline=1500.0):
    """All completed matches for `team_name`, one row per appearance (home or away),
    expanded into the long-form schema training uses for rolling features.
    Sorted ascending by MatchDate. Mirrors training.py home_records+away_records.
    """
    if matches_df is None or len(matches_df) == 0:
        return pd.DataFrame()

    def _col(df, name, default=0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    home_m = matches_df[matches_df['HomeTeam'] == team_name]
    away_m = matches_df[matches_df['AwayTeam'] == team_name]

    def _expand(df, role):
        if len(df) == 0:
            return pd.DataFrame()
        is_home = (role == 'home')
        gs = df['FTHome'] if is_home else df['FTAway']
        gc = df['FTAway'] if is_home else df['FTHome']
        pts_map = {'H': 3, 'D': 1, 'A': 0} if is_home else {'H': 0, 'D': 1, 'A': 3}
        won_marker = 'H' if is_home else 'A'
        shots_target = _col(df, 'HomeTarget' if is_home else 'AwayTarget')
        shots       = _col(df, 'HomeShots'  if is_home else 'AwayShots')
        target      = shots_target
        # xG proxy: 0.30 per on-target shot + 0.05 per off-target shot — must match training
        xg = (shots_target.fillna(0) * 0.30
              + (shots.fillna(0) - shots_target.fillna(0)).clip(lower=0) * 0.05)
        corners     = _col(df, 'HomeCorners' if is_home else 'AwayCorners')
        fouls       = _col(df, 'HomeFouls'   if is_home else 'AwayFouls')
        yellow      = _col(df, 'HomeYellow'  if is_home else 'AwayYellow')
        red         = _col(df, 'HomeRed'     if is_home else 'AwayRed')
        form3       = _col(df, 'Form3Home'   if is_home else 'Form3Away')
        form5       = _col(df, 'Form5Home'   if is_home else 'Form5Away')
        # Opponent Elo for this match (away Elo if team played at home, vice versa)
        opp_elo = _col(df, 'AwayElo' if is_home else 'HomeElo', default=elo_baseline)
        # Multiplier matches training's clip(0.5, 1.5)
        elo_mult = (opp_elo.fillna(elo_baseline) / elo_baseline).clip(0.5, 1.5)
        points_series = df['FTResult'].map(pts_map).fillna(0)
        goal_diff_series = (gs - gc).fillna(0)
        gs_filled = gs.fillna(0)
        gc_filled = gc.fillna(0)
        xg_filled = xg.fillna(0)
        return pd.DataFrame({
            'MatchDate':     df['MatchDate'],
            'Side':          role,
            'GoalsScored':   gs,
            'GoalsConceded': gc,
            'GoalDiff':      gs - gc,
            'Points':        df['FTResult'].map(pts_map),
            'Won':           (df['FTResult'] == won_marker).astype(int),
            'IsDraw':        (df['FTResult'] == 'D').astype(int),
            'ShotsOnTarget': shots_target,
            'Shots':         shots,
            'Target':        target,
            'XG':            xg,
            'Corners':       corners,
            'Fouls':         fouls,
            'Yellow':        yellow,
            'Red':           red,
            'Form3':         form3,
            'Form5':         form5,
            # Opponent-Elo-weighted per-match stats — drive _EloWtd_Last5/10 rolling features
            'GoalsScored_EloWtd':   gs_filled * elo_mult,
            'GoalsConceded_EloWtd': gc_filled * elo_mult,
            'Points_EloWtd':        points_series * elo_mult,
            'GoalDiff_EloWtd':      goal_diff_series * elo_mult,
            'XG_EloWtd':            xg_filled * elo_mult,
        })

    long = pd.concat([_expand(home_m, 'home'), _expand(away_m, 'away')], ignore_index=True)
    if len(long) == 0:
        return long
    long = long.dropna(subset=['MatchDate']).sort_values('MatchDate').reset_index(drop=True)
    return long


def get_team_recent_stats(team_name, matches_df, prediction_date=None, elo_baseline=1500.0):
    """Predict-time analogue of training's rolling features.

    Training computes `x.shift(1).rolling(N).mean()` per team — for the "next" match,
    this reduces to mean over the team's most recent N matches. Same logic here.

    Returns:
        {
          'Last5':         {stat: mean over last 5},  # ROLLING_STATS
          'Last10':        {stat: mean over last 10},
          'WinStreak':     trailing consecutive wins,
          'DaysSinceLast': days from prediction_date to most recent match,
          'Recent5':       {stat: mean over last 5},  # LEGACY_STATS for HomeShots/etc.
          'ShotConversion': Target/Shots over last 5 (avoids div-by-zero),
        }
    """
    history = _team_long_history(team_name, matches_df, elo_baseline=elo_baseline)
    empty = {
        'Last5':         {s: 0.0 for s in ROLLING_STATS},
        'Last10':        {s: 0.0 for s in ROLLING_STATS},
        'WinStreak':     0,
        'DaysSinceLast': 14.0,
        'Recent5':       {s: 0.0 for s in LEGACY_STATS},
        'ShotConversion': 0.0,
    }
    if len(history) == 0:
        return empty

    if prediction_date is None:
        prediction_date = pd.Timestamp.now()

    last5_df  = history.tail(5)
    last10_df = history.tail(10)

    # EWM mean over the last N — mirrors training's shift(1).ewm(span=N).mean() at the "next match".
    # For predict-time we evaluate at the row after the most recent match, so shift is implicit.
    def _ewm_mean(series, span):
        return float(series.fillna(0).ewm(span=span, min_periods=1).mean().iloc[-1]) if len(series) else 0.0

    last5  = {s: _ewm_mean(last5_df[s], 5)   if s in last5_df.columns  else 0.0 for s in ROLLING_STATS}
    last10 = {s: _ewm_mean(last10_df[s], 10) if s in last10_df.columns else 0.0 for s in ROLLING_STATS}
    # Recent5 keeps plain mean — used for per-match features like HomeShots where training
    # uses the raw match value (no smoothing intended).
    recent5 = {s: float(last5_df[s].fillna(0).mean()) if s in last5_df.columns else 0.0 for s in LEGACY_STATS}

    # Trailing-wins streak: count consecutive 1s from the end (matches training's calc_streak).
    streak = 0
    for v in reversed(history['Won'].fillna(0).tolist()):
        if v == 1:
            streak += 1
        else:
            break

    last_date = history['MatchDate'].iloc[-1]
    days_since_last = (prediction_date - last_date).days if pd.notna(last_date) else 14
    if days_since_last is None or days_since_last < 0 or pd.isna(days_since_last):
        days_since_last = 14

    if recent5.get('Shots', 0) > 0:
        shot_conv = recent5.get('Target', 0) / recent5['Shots']
    else:
        shot_conv = 0.0

    return {
        'Last5':          last5,
        'Last10':         last10,
        'WinStreak':      streak,
        'DaysSinceLast':  float(days_since_last),
        'Recent5':        recent5,
        'ShotConversion': float(shot_conv),
    }


def get_team_side_form(team_name, matches_df, side, windows=(5, 10)):
    """Rolling Points/GoalDiff over team's matches played as `side` only — mirrors
    training's compute_side_form. side ∈ {'home', 'away'}.
    """
    out = {f'Last{w}': {'Points': 0.0, 'GoalDiff': 0.0} for w in windows}
    history = _team_long_history(team_name, matches_df)
    if len(history) == 0:
        return out
    side_only = history[history['Side'] == side]
    for w in windows:
        recent = side_only.tail(w)
        if len(recent) == 0:
            continue
        # EWM matches training's shift(1).ewm(span=w).mean() at the next match.
        out[f'Last{w}'] = {
            'Points':   float(recent['Points'].fillna(0).ewm(span=w, min_periods=1).mean().iloc[-1]),
            'GoalDiff': float(recent['GoalDiff'].fillna(0).ewm(span=w, min_periods=1).mean().iloc[-1]),
        }
    return out


def get_h2h_stats(home_team, away_team, matches_df):
    """H2H stats relative to current home team's perspective across all prior pair-meetings.
    Mirrors training's compute_h2h_features but over the full history (no shift needed
    since the hypothetical match isn't in matches_df).
    """
    empty = {'H2H_HomeWinRate': 0.0, 'H2H_DrawRate': 0.0,
             'H2H_TotalMatches': 0.0, 'H2H_AvgGoalDiff': 0.0}
    if matches_df is None or len(matches_df) == 0:
        return empty
    pair = matches_df[
        ((matches_df['HomeTeam'] == home_team) & (matches_df['AwayTeam'] == away_team)) |
        ((matches_df['HomeTeam'] == away_team) & (matches_df['AwayTeam'] == home_team))
    ].dropna(subset=['MatchDate']).sort_values('MatchDate')
    n = len(pair)
    if n == 0:
        return empty
    wins, draws, gd_sum = 0, 0, 0.0
    for _, row in pair.iterrows():
        res = row['FTResult']
        if res == 'D':
            draws += 1
            continue
        if row['HomeTeam'] == home_team:
            if res == 'H':
                wins += 1
            gd_sum += (row['FTHome'] - row['FTAway'])
        else:
            if res == 'A':
                wins += 1
            gd_sum += (row['FTAway'] - row['FTHome'])
    return {
        'H2H_HomeWinRate':  wins / n,
        'H2H_DrawRate':     draws / n,
        'H2H_TotalMatches': float(n),
        'H2H_AvgGoalDiff':  gd_sum / n,
    }


def _pick_odds(latest, base_keys, max_keys):
    """Return triple of odds from `latest`, preferring Max* (training parity)."""
    if latest is None:
        return None, None, None
    def _val(k):
        if k in latest and pd.notna(latest[k]):
            return float(latest[k])
        return None
    base = [_val(k) for k in base_keys]
    if max_keys is not None:
        for i, mk in enumerate(max_keys):
            mv = _val(mk)
            if mv is not None:
                base[i] = mv
    return tuple(base)


def _resolve_division(home_team, matches_df, latest_match):
    """Best-effort Division for this matchup."""
    if latest_match is not None and 'Division' in latest_match.index and pd.notna(latest_match['Division']):
        return latest_match['Division']
    if matches_df is not None and 'Division' in matches_df.columns:
        home_recent = matches_df[matches_df['HomeTeam'] == home_team].dropna(subset=['MatchDate']).sort_values('MatchDate')
        if len(home_recent) > 0:
            d = home_recent.iloc[-1].get('Division')
            if pd.notna(d):
                return d
    return None


def build_feature_vector(home_team, away_team, matches_df, elo_df, feature_columns, elo_baseline=1500.0):
    home_elo = get_team_elo(home_team, elo_df, matches_df, is_home=True)
    away_elo = get_team_elo(away_team, elo_df, matches_df, is_home=False)
    prediction_date = pd.Timestamp.now()
    home_stats = get_team_recent_stats(home_team, matches_df, prediction_date, elo_baseline=elo_baseline)
    away_stats = get_team_recent_stats(away_team, matches_df, prediction_date, elo_baseline=elo_baseline)
    home_side_form = get_team_side_form(home_team, matches_df, side='home')
    away_side_form = get_team_side_form(away_team, matches_df, side='away')
    h2h = get_h2h_stats(home_team, away_team, matches_df)
    elo_diff = home_elo - away_elo
    elo_total = home_elo + away_elo

    # Latest matchup (for stale odds + division context)
    latest_match = None
    if matches_df is not None:
        matchup = matches_df[
            (matches_df['HomeTeam'] == home_team) & (matches_df['AwayTeam'] == away_team)
        ].dropna(subset=['MatchDate']).sort_values('MatchDate')
        if len(matchup) > 0:
            latest_match = matchup.iloc[-1]

    # 1X2 market — prefer MaxHome/MaxDraw/MaxAway, fall back to OddHome/OddDraw/OddAway
    odd_h, odd_d, odd_a = _pick_odds(
        latest_match, ['OddHome', 'OddDraw', 'OddAway'],
        ['MaxHome', 'MaxDraw', 'MaxAway']
    )
    impl_home = 1.0 / odd_h if odd_h else 0.33
    impl_draw = 1.0 / odd_d if odd_d else 0.33
    impl_away = 1.0 / odd_a if odd_a else 0.33
    _margin = impl_home + impl_draw + impl_away
    impl_home /= _margin; impl_draw /= _margin; impl_away /= _margin

    # Over/Under 2.5 — prefer MaxOver25/MaxUnder25, fall back to Over25/Under25
    over25, under25 = _pick_odds(
        latest_match, ['Over25', 'Under25'], ['MaxOver25', 'MaxUnder25']
    )[:2] if latest_match is not None else (None, None)
    if over25 and under25:
        ip_o = 1.0 / over25
        ip_u = 1.0 / under25
        _ou = ip_o + ip_u
        impl_over25 = ip_o / _ou
        impl_under25 = ip_u / _ou
    else:
        impl_over25, impl_under25 = 0.5, 0.5
    over25_diff = impl_over25 - impl_under25

    # Asian Handicap
    handi_size, handi_impl_home, handi_impl_away = 0.0, 0.5, 0.5
    if latest_match is not None:
        hh = latest_match.get('HandiHome') if 'HandiHome' in latest_match.index else None
        ha = latest_match.get('HandiAway') if 'HandiAway' in latest_match.index else None
        hs = latest_match.get('HandiSize') if 'HandiSize' in latest_match.index else None
        if pd.notna(hs):
            handi_size = float(hs)
        if pd.notna(hh) and pd.notna(ha) and hh and ha:
            ih = 1.0 / float(hh); ia = 1.0 / float(ha)
            _hm = ih + ia
            handi_impl_home = ih / _hm
            handi_impl_away = ia / _hm

    division_for_match = _resolve_division(home_team, matches_df, latest_match)

    h_last5,  h_last10  = home_stats['Last5'], home_stats['Last10']
    a_last5,  a_last10  = away_stats['Last5'], away_stats['Last10']
    h_side5,  h_side10  = home_side_form['Last5'], home_side_form['Last10']
    a_side5,  a_side10  = away_side_form['Last5'], away_side_form['Last10']
    h_recent5 = home_stats['Recent5']
    a_recent5 = away_stats['Recent5']

    rest_diff   = home_stats['DaysSinceLast'] - away_stats['DaysSinceLast']
    streak_diff = home_stats['WinStreak']     - away_stats['WinStreak']

    # Rolling family lookup tables
    rolling_lookup = {  # Home_/Away_ prefix
        'Home': (h_last5, h_last10),
        'Away': (a_last5, a_last10),
    }
    side_form_lookup = {
        'HomeForm_AtHome': (h_side5, h_side10),
        'AwayForm_AtAway': (a_side5, a_side10),
    }

    unmatched = []
    features = []
    for col in feature_columns:
        val = None

        # ── direct named features ──
        if   col == 'HomeElo':              val = home_elo
        elif col == 'AwayElo':              val = away_elo
        elif col == 'EloDifference':        val = elo_diff
        elif col == 'EloTotal':             val = elo_total
        elif col == 'Form3Home':            val = h_recent5.get('Form3', 0)
        elif col == 'Form5Home':            val = h_recent5.get('Form5', 0)
        elif col == 'Form3Away':            val = a_recent5.get('Form3', 0)
        elif col == 'Form5Away':            val = a_recent5.get('Form5', 0)
        elif col == 'Form3Diff':            val = h_recent5.get('Form3', 0) - a_recent5.get('Form3', 0)
        elif col == 'Form5Diff':            val = h_recent5.get('Form5', 0) - a_recent5.get('Form5', 0)
        elif col == 'HomeShots':            val = h_recent5.get('Shots', 0)
        elif col == 'AwayShots':            val = a_recent5.get('Shots', 0)
        elif col == 'ShotsDifference':      val = h_recent5.get('Shots', 0) - a_recent5.get('Shots', 0)
        elif col == 'HomeXG':               val = h_recent5.get('XG', 0)
        elif col == 'AwayXG':               val = a_recent5.get('XG', 0)
        elif col == 'XGDiff':               val = h_recent5.get('XG', 0) - a_recent5.get('XG', 0)
        elif col == 'HomeTarget':           val = h_recent5.get('Target', 0)
        elif col == 'AwayTarget':           val = a_recent5.get('Target', 0)
        elif col == 'HomeCorners':          val = h_recent5.get('Corners', 0)
        elif col == 'AwayCorners':          val = a_recent5.get('Corners', 0)
        elif col == 'CornersDifference':    val = h_recent5.get('Corners', 0) - a_recent5.get('Corners', 0)
        elif col == 'HomeFouls':            val = h_recent5.get('Fouls', 0)
        elif col == 'AwayFouls':            val = a_recent5.get('Fouls', 0)
        elif col == 'HomeYellow':           val = h_recent5.get('Yellow', 0)
        elif col == 'AwayYellow':           val = a_recent5.get('Yellow', 0)
        elif col == 'HomeRed':              val = h_recent5.get('Red', 0)
        elif col == 'AwayRed':              val = a_recent5.get('Red', 0)
        # 1X2 market
        elif col == 'ImpliedProbHome':      val = impl_home
        elif col == 'ImpliedProbDraw':      val = impl_draw
        elif col == 'ImpliedProbAway':      val = impl_away
        elif col == 'OddsDiffHomeAway':     val = impl_home - impl_away
        # Over/Under 2.5
        elif col == 'ImpliedProbOver25':    val = impl_over25
        elif col == 'ImpliedProbUnder25':   val = impl_under25
        elif col == 'Over25Diff':           val = over25_diff
        # Asian Handicap
        elif col == 'HandiSizeNum':         val = handi_size
        elif col == 'HandiImpliedHome':     val = handi_impl_home
        elif col == 'HandiImpliedAway':     val = handi_impl_away
        # Shot conversion
        elif col == 'HomeShotConversion':   val = home_stats.get('ShotConversion', 0)
        elif col == 'AwayShotConversion':   val = away_stats.get('ShotConversion', 0)
        elif col == 'ShotConversionDiff':   val = home_stats.get('ShotConversion', 0) - away_stats.get('ShotConversion', 0)
        # Rest / streak diffs (rolling family but training emits them as flat names)
        elif col == 'Home_WinStreak':       val = home_stats['WinStreak']
        elif col == 'Away_WinStreak':       val = away_stats['WinStreak']
        elif col == 'WinStreakDiff':        val = streak_diff
        elif col == 'Home_DaysSinceLast':   val = home_stats['DaysSinceLast']
        elif col == 'Away_DaysSinceLast':   val = away_stats['DaysSinceLast']
        elif col == 'RestDiff':             val = rest_diff
        # H2H
        elif col == 'H2H_HomeWinRate':      val = h2h['H2H_HomeWinRate']
        elif col == 'H2H_DrawRate':         val = h2h['H2H_DrawRate']
        elif col == 'H2H_TotalMatches':     val = h2h['H2H_TotalMatches']
        elif col == 'H2H_AvgGoalDiff':      val = h2h['H2H_AvgGoalDiff']
        # GoalsScored rolling diff
        elif col == 'GoalsScored_Diff_Last5':
            val = h_last5.get('GoalsScored', 0) - a_last5.get('GoalsScored', 0)
        elif col == 'GoalsScored_Diff_Last10':
            val = h_last10.get('GoalsScored', 0) - a_last10.get('GoalsScored', 0)
        # Opponent-Elo-weighted rolling diffs
        elif col.endswith('_EloWtd_Diff_Last5') or col.endswith('_EloWtd_Diff_Last10'):
            window = '5' if col.endswith('_Last5') else '10'
            stat = col[: -len(f'_Diff_Last{window}')]  # e.g. 'GoalsScored_EloWtd'
            src_h = h_last5 if window == '5' else h_last10
            src_a = a_last5 if window == '5' else a_last10
            val = src_h.get(stat, 0) - src_a.get(stat, 0)
        # Division one-hot
        elif col.startswith('Div_'):
            val = 1 if (division_for_match is not None and col == f'Div_{division_for_match}') else 0
        else:
            # ── rolling family by suffix ──
            handled = False
            for prefix, (l5, l10) in rolling_lookup.items():
                pfx = f'{prefix}_'
                if col.startswith(pfx) and (col.endswith('_Last5') or col.endswith('_Last10')):
                    stat = col[len(pfx):].rsplit('_Last', 1)[0]
                    window = col.rsplit('_Last', 1)[1]
                    src = l5 if window == '5' else l10
                    val = src.get(stat, 0)
                    handled = True
                    break
            if not handled:
                # ── side-form family (HomeForm_AtHome_/AwayForm_AtAway_) ──
                for prefix, (s5, s10) in side_form_lookup.items():
                    pfx = f'{prefix}_'
                    if col.startswith(pfx) and (col.endswith('_Last5') or col.endswith('_Last10')):
                        stat = col[len(pfx):].rsplit('_Last', 1)[0]
                        window = col.rsplit('_Last', 1)[1]
                        src = s5 if window == '5' else s10
                        val = src.get(stat, 0)
                        handled = True
                        break
            if not handled:
                unmatched.append(col)
                val = 0

        features.append(0 if val is None else float(val))

    if unmatched:
        logger.warning("Unmatched feature columns (zero-filled): %s", unmatched[:10])
        print(f"Unmatched feature columns (zero-filled): {unmatched[:10]}")

    context = {
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'home_side_form': home_side_form,
        'away_side_form': away_side_form,
        'h2h': h2h,
    }
    return np.array([features]), home_elo, away_elo, context


# ──────────────────────────────────────────────
# 🗣️  REASONING (fan-friendly explanations)
# ──────────────────────────────────────────────

def _join_list(items):
    """Join strings as readable English: 'a, b and c'."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _factor_phrase_for_winner(factor, target_side):
    """Short phrase used inside the summary to describe a factor supporting the winner."""
    fid = factor['id']
    is_home = target_side == 'home'
    winner_val = factor['home_value'] if is_home else factor['away_value']
    loser_val = factor['away_value'] if is_home else factor['home_value']
    if fid == 'elo':
        return f"a better Elo rating ({winner_val:.0f} vs {loser_val:.0f})"
    if fid == 'form_last5':
        return f"stronger recent form ({winner_val:.2f} vs {loser_val:.2f} pts/match over the last 5)"
    if fid == 'side_form':
        where = "at home" if is_home else "on the road"
        return f"a stronger record {where} ({winner_val:.2f} pts/match)"
    if fid == 'h2h':
        return f"a favorable head-to-head record ({int(winner_val)}-{int(loser_val)})"
    if fid == 'goal_diff_last5':
        return f"a better recent goal difference ({winner_val:+.2f} vs {loser_val:+.2f} goals/match)"
    if fid == 'win_streak':
        return f"a {int(winner_val)}-match win streak"
    return factor['label'].lower()


def _opposing_caveat(factor, opp_team):
    """One-sentence caveat acknowledging the strongest signal against the predicted winner."""
    fid = factor['id']
    if fid == 'elo':
        return f"That said, {opp_team} has a higher Elo rating."
    if fid == 'form_last5':
        return f"However, {opp_team} comes in with better form over the last 5 matches."
    if fid == 'side_form':
        return f"That said, {opp_team} has a stronger record on its own turf."
    if fid == 'h2h':
        return f"However, the head-to-head history favors {opp_team}."
    if fid == 'goal_diff_last5':
        return f"That said, {opp_team} has a better recent goal difference."
    if fid == 'win_streak':
        return f"That said, {opp_team} is on a win streak of its own."
    return ""


def build_reasoning(prediction, home_team, away_team, context):
    """Build a human-readable explanation (Romanian) of why the model picked this outcome.

    Returns a dict with `summary` (string) and `factors` (list of structured items the UI can
    render as a table/bar chart). This is a *narrative consistent with* the prediction — it
    surfaces the fan-relevant signals (Elo, recent form, H2H, side form, goal diff, streak)
    and says which side each favors. It is NOT a decomposition of the meta-learner's output.
    """
    home_elo = context['home_elo']
    away_elo = context['away_elo']
    home_stats = context['home_stats']
    away_stats = context['away_stats']
    home_side_form = context['home_side_form']
    away_side_form = context['away_side_form']
    h2h = context['h2h']

    factors = []

    # 1. Elo
    elo_diff = home_elo - away_elo
    elo_favors = 'home' if elo_diff > 25 else ('away' if elo_diff < -25 else 'none')
    elo_desc = f"{home_team} Elo: {home_elo:.0f}, {away_team} Elo: {away_elo:.0f}."
    if elo_favors != 'none':
        winner_name = home_team if elo_favors == 'home' else away_team
        elo_desc += f" {winner_name} leads by {abs(elo_diff):.0f} points."
    else:
        elo_desc += " Negligible difference."
    factors.append({
        'id': 'elo', 'label': 'Elo rating',
        'home_value': round(home_elo, 0), 'away_value': round(away_elo, 0),
        'favors': elo_favors, 'strength': abs(elo_diff) / 200.0,
        'description': elo_desc,
    })

    # 2. Form — points/match over last 5
    h_pts5 = home_stats['Last5'].get('Points', 0)
    a_pts5 = away_stats['Last5'].get('Points', 0)
    pts_diff = h_pts5 - a_pts5
    form_favors = 'home' if pts_diff > 0.3 else ('away' if pts_diff < -0.3 else 'none')
    form_desc = (
        f"{home_team} averaged {h_pts5:.2f} pts/match, "
        f"{away_team} {a_pts5:.2f} pts/match over the last 5."
    )
    factors.append({
        'id': 'form_last5', 'label': 'Form (last 5 matches)',
        'home_value': round(h_pts5, 2), 'away_value': round(a_pts5, 2),
        'favors': form_favors, 'strength': abs(pts_diff),
        'description': form_desc,
    })

    # 3. Side form — home at home vs away on the road
    h_home_pts = home_side_form['Last5'].get('Points', 0)
    a_away_pts = away_side_form['Last5'].get('Points', 0)
    side_diff = h_home_pts - a_away_pts
    side_favors = 'home' if side_diff > 0.3 else ('away' if side_diff < -0.3 else 'none')
    side_desc = (
        f"{home_team} at home: {h_home_pts:.2f} pts/match. "
        f"{away_team} away: {a_away_pts:.2f} pts/match."
    )
    factors.append({
        'id': 'side_form', 'label': 'Home / away record',
        'home_value': round(h_home_pts, 2), 'away_value': round(a_away_pts, 2),
        'favors': side_favors, 'strength': abs(side_diff) * 0.8,
        'description': side_desc,
    })

    # 4. H2H (skip if too few meetings)
    n_h2h = int(h2h.get('H2H_TotalMatches', 0))
    if n_h2h >= 3:
        rate = h2h['H2H_HomeWinRate']
        draw_rate = h2h['H2H_DrawRate']
        home_w = int(round(rate * n_h2h))
        draws = int(round(draw_rate * n_h2h))
        away_w = max(0, n_h2h - home_w - draws)
        if home_w >= away_w + 2:
            h2h_favors = 'home'
        elif away_w >= home_w + 2:
            h2h_favors = 'away'
        else:
            h2h_favors = 'none'
        factors.append({
            'id': 'h2h', 'label': 'Head-to-head',
            'home_value': float(home_w), 'away_value': float(away_w),
            'favors': h2h_favors, 'strength': abs(home_w - away_w) / max(n_h2h, 1) * 1.2,
            'description': (
                f"Over {n_h2h} head-to-head matches: {home_w} wins for {home_team}, "
                f"{draws} draws, {away_w} wins for {away_team}."
            ),
        })

    # 5. Goal difference last 5
    h_gd5 = home_stats['Last5'].get('GoalDiff', 0)
    a_gd5 = away_stats['Last5'].get('GoalDiff', 0)
    gd_diff = h_gd5 - a_gd5
    gd_favors = 'home' if gd_diff > 0.3 else ('away' if gd_diff < -0.3 else 'none')
    factors.append({
        'id': 'goal_diff_last5', 'label': 'Goal difference (last 5)',
        'home_value': round(h_gd5, 2), 'away_value': round(a_gd5, 2),
        'favors': gd_favors, 'strength': abs(gd_diff) * 0.6,
        'description': (
            f"{home_team}: {h_gd5:+.2f} goals/match, "
            f"{away_team}: {a_gd5:+.2f} goals/match."
        ),
    })

    # 6. Win streak — only emit if at least one team is on a real run
    h_streak = int(home_stats.get('WinStreak', 0))
    a_streak = int(away_stats.get('WinStreak', 0))
    if max(h_streak, a_streak) >= 3:
        if h_streak - a_streak >= 2:
            streak_favors = 'home'
        elif a_streak - h_streak >= 2:
            streak_favors = 'away'
        else:
            streak_favors = 'none'
        factors.append({
            'id': 'win_streak', 'label': 'Win streak',
            'home_value': float(h_streak), 'away_value': float(a_streak),
            'favors': streak_favors, 'strength': abs(h_streak - a_streak) * 0.5,
            'description': (
                f"{home_team}: {h_streak} consecutive wins. "
                f"{away_team}: {a_streak} consecutive wins."
            ),
        })

    # ── English summary ──
    pred = int(prediction)
    if pred == 0:
        summary = (
            "The model expects an evenly matched game. The key signals "
            "(Elo, recent form, head-to-head) do not clearly favor either side."
        )
    else:
        target_side = 'home' if pred == 1 else 'away'
        winner = home_team if pred == 1 else away_team
        location = "at home" if pred == 1 else "on the road"

        supporting = sorted(
            (f for f in factors if f['favors'] == target_side),
            key=lambda f: -f['strength'],
        )
        opposing = sorted(
            (f for f in factors if f['favors'] not in (target_side, 'none')),
            key=lambda f: -f['strength'],
        )

        if supporting:
            phrases = [_factor_phrase_for_winner(f, target_side) for f in supporting[:3]]
            summary = f"{winner} is favored {location} thanks to {_join_list(phrases)}."
        else:
            summary = (
                f"The model picks {winner} {location} without a single dominant fundamental signal — "
                f"the call rests on the finer combination of indicators."
            )

        if opposing:
            opp = opposing[0]
            opp_team = home_team if opp['favors'] == 'home' else away_team
            caveat = _opposing_caveat(opp, opp_team)
            if caveat:
                summary += " " + caveat

    return {
        'summary': summary,
        'factors': [
            {k: v for k, v in f.items() if k != 'strength'}
            for f in factors
        ],
    }


# ──────────────────────────────────────────────
# 🚀 ENDPOINTS
# ──────────────────────────────────────────────

@router.post("/", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """
    Predict the outcome of a football match.
    """
    try:
        logger.info("Prediction request: %s vs %s", request.home_team, request.away_team)
        print(f"Prediction request: {request.home_team} vs {request.away_team}")
        rf_model, lr_model, xgb_model, le, feature_columns, meta_learner, draw_model = load_models()

        if rf_model is None or feature_columns is None:
            logger.warning("Models not loaded - returning 503")
            print("Models not loaded - returning 503")
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded! Run /train first."
            )

        logger.info("Loading dataset from Kaggle...")
        print("Loading dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
            matches_df = pd.read_csv(os.path.join(path, 'Matches.csv'))
            matches_df['MatchDate'] = pd.to_datetime(matches_df['MatchDate'], errors='coerce')
            matches_df = matches_df.sort_values('MatchDate')
            logger.info("Matches loaded: %d rows", len(matches_df))
            print(f"Matches loaded: {len(matches_df)} rows")
        except Exception:
            logger.warning("Could not load Matches.csv")
            print("Could not load Matches.csv")
            matches_df = None

        try:
            elo_df = pd.read_csv(os.path.join(path, 'EloRatings.csv') if matches_df is not None else None)
            elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
            logger.info("Elo ratings loaded: %d rows", len(elo_df))
            print(f"Elo ratings loaded: {len(elo_df)} rows")
        except Exception:
            logger.warning("Could not load EloRatings.csv")
            print("Could not load EloRatings.csv")
            elo_df = None

        logger.info("Building feature vector...")
        print("Building feature vector...")
        elo_baseline = load_elo_baseline()
        features, home_elo, away_elo, context = build_feature_vector(
            request.home_team, request.away_team, matches_df, elo_df, feature_columns,
            elo_baseline=elo_baseline,
        )

        logger.info("Running prediction (HomeElo=%.1f, AwayElo=%.1f)", home_elo, away_elo)
        print(f"Running prediction (HomeElo={home_elo:.1f}, AwayElo={away_elo:.1f})")

        # Get probabilities from all 3 base models
        rf_proba = rf_model.predict_proba(features)[0]    # classes: [-1, 0, 1]
        lr_proba = lr_model.predict_proba(features)[0]    # classes: [-1, 0, 1]
        xgb_proba_raw = xgb_model.predict_proba(features)[0]  # classes: encoded

        # Reorder XGB proba to match [-1, 0, 1] order
        xgb_class_order = le.inverse_transform(xgb_model.classes_)
        target_order = sorted(rf_model.classes_)
        xgb_col_map = [list(xgb_class_order).index(c) for c in target_order]
        xgb_proba = xgb_proba_raw[xgb_col_map]

        if meta_learner is not None:
            # Stacking: meta-learner consumes the 10-column matrix [rf, lr, xgb, draw_p].
            # Layout must match training: rf_p (3) + lr_p (3) + xgb_p (3) + draw_p (1).
            if draw_model is not None:
                draw_p = float(draw_model.predict_proba(features)[0, 1])
            else:
                # Missing draw model — feed 0 so the meta-learner just ignores that column.
                draw_p = 0.0
            meta_row = np.hstack([rf_proba, lr_proba, xgb_proba, [draw_p]]).reshape(1, -1)
            ensemble_proba = meta_learner.predict_proba(meta_row)[0]
            target_order = list(meta_learner.classes_)  # already sorted [-1, 0, 1]
            prediction = target_order[int(np.argmax(ensemble_proba))]
        else:
            # Fallback for older model pickles without meta-learner — fixed legacy weights.
            logger.warning("meta_learner.pkl missing — falling back to legacy 0.4/0.35/0.25 blend")
            print("meta_learner.pkl missing — falling back to legacy 0.4/0.35/0.25 blend")
            ensemble_proba = 0.4 * xgb_proba + 0.35 * rf_proba + 0.25 * lr_proba
            prediction = target_order[int(np.argmax(ensemble_proba))]

        prob_dict = {}
        for i, label in enumerate(target_order):
            prob_dict[str(int(label))] = ensemble_proba[i] * 100

        result_map = {
            1.0:  "🏠 HOME win",
            0.0:  "🤝 DRAW",
            -1.0: "🚗 AWAY win"
        }

        home_win_prob = prob_dict.get("1", 0)
        draw_prob = prob_dict.get("0", 0)
        away_win_prob = prob_dict.get("-1", 0)

        reasoning = build_reasoning(prediction, request.home_team, request.away_team, context)

        logger.info("Prediction result: %s -> %s (confidence: %.1f%%)",
                    f"{request.home_team} vs {request.away_team}",
                    result_map.get(prediction, "???"), max(ensemble_proba) * 100)
        print(f"Prediction result: {request.home_team} vs {request.away_team} -> "
              f"{result_map.get(prediction, '???')} (confidence: {max(ensemble_proba) * 100:.1f}%)")

        return {
            "match": f"{request.home_team} vs {request.away_team}",
            "prediction": result_map.get(prediction, "❓ ???"),
            "confidence": max(ensemble_proba) * 100,
            "home_team": request.home_team,
            "away_team": request.away_team,
            "home_elo": round(home_elo, 2),
            "away_elo": round(away_elo, 2),
            "elo_difference": round(home_elo - away_elo, 2),
            "probabilities": prob_dict,
            "home_win_prob": round(home_win_prob, 2),
            "draw_prob": round(draw_prob, 2),
            "away_win_prob": round(away_win_prob, 2),
            "reasoning": reasoning,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )
