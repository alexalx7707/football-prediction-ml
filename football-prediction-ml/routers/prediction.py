"""
Router for match prediction endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
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
    home_team: str = Field(..., description="Echipa de acasă")
    away_team: str = Field(..., description="Echipa din deplasare")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Manchester United",
                "away_team": "Liverpool"
            }
        }


class PredictionResponse(BaseModel):
    match: str
    prediction: str
    confidence: float = Field(..., description="Procentajul de încredere (0-100)")
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    elo_difference: float
    probabilities: Dict[str, float]
    home_win_prob: float
    draw_prob: float
    away_win_prob: float


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
        return rf_model, lr_model, xgb_model, le, feature_columns
    except Exception:
        return None, None, None, None, None


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


def get_team_recent_stats(team_name, matches_df, n=5):
    if matches_df is None:
        return {}
    home_matches = matches_df[matches_df['HomeTeam'] == team_name].copy()
    home_matches = home_matches.rename(columns={
        'HomeShots': 'Shots', 'HomeTarget': 'Target', 'HomeCorners': 'Corners',
        'HomeFouls': 'Fouls', 'HomeYellow': 'Yellow', 'HomeRed': 'Red',
        'Form3Home': 'Form3', 'Form5Home': 'Form5'
    })
    home_matches['GoalsScored'] = matches_df.loc[home_matches.index, 'FTHome']
    home_matches['GoalsConceded'] = matches_df.loc[home_matches.index, 'FTAway']
    home_matches['Points'] = matches_df.loc[home_matches.index, 'FTResult'].map({'H': 3, 'D': 1, 'A': 0})
    home_matches['Won'] = (matches_df.loc[home_matches.index, 'FTResult'] == 'H').astype(int)
    home_matches['GoalDiff'] = home_matches['GoalsScored'] - home_matches['GoalsConceded']
    away_matches = matches_df[matches_df['AwayTeam'] == team_name].copy()
    away_matches = away_matches.rename(columns={
        'AwayShots': 'Shots', 'AwayTarget': 'Target', 'AwayCorners': 'Corners',
        'AwayFouls': 'Fouls', 'AwayYellow': 'Yellow', 'AwayRed': 'Red',
        'Form3Away': 'Form3', 'Form5Away': 'Form5'
    })
    away_matches['GoalsScored'] = matches_df.loc[away_matches.index, 'FTAway']
    away_matches['GoalsConceded'] = matches_df.loc[away_matches.index, 'FTHome']
    away_matches['Points'] = matches_df.loc[away_matches.index, 'FTResult'].map({'H': 0, 'D': 1, 'A': 3})
    away_matches['Won'] = (matches_df.loc[away_matches.index, 'FTResult'] == 'A').astype(int)
    away_matches['GoalDiff'] = away_matches['GoalsScored'] - away_matches['GoalsConceded']
    stat_cols = ['MatchDate', 'Shots', 'Target', 'Corners', 'Fouls', 'Yellow', 'Red', 'Form3', 'Form5',
                 'GoalsScored', 'GoalsConceded', 'Points', 'Won', 'GoalDiff']
    existing_home = [c for c in stat_cols if c in home_matches.columns]
    existing_away = [c for c in stat_cols if c in away_matches.columns]
    all_matches = pd.concat([
        home_matches[existing_home],
        away_matches[existing_away]
    ]).sort_values('MatchDate')
    if len(all_matches) == 0:
        return {}
    recent = all_matches.tail(n)
    stats = {}
    for col in ['Shots', 'Target', 'Corners', 'Fouls', 'Yellow', 'Red', 'Form3', 'Form5',
                'GoalsScored', 'GoalsConceded', 'Points', 'GoalDiff']:
        if col in recent.columns:
            stats[col] = recent[col].fillna(0).mean()
    if 'Target' in stats and 'Shots' in stats and stats['Shots'] > 0:
        stats['ShotConversion'] = stats['Target'] / stats['Shots']
    else:
        stats['ShotConversion'] = 0
    # Win streak from most recent games
    all_recent = all_matches.tail(10)
    streak = 0
    if len(all_recent) > 0 and 'Won' in all_recent.columns:
        for val in reversed(all_recent['Won'].fillna(0).tolist()):
            if val == 1:
                streak += 1
            else:
                break
    stats['WinStreak'] = streak
    return stats


def build_feature_vector(home_team, away_team, matches_df, elo_df, feature_columns):
    home_elo = get_team_elo(home_team, elo_df, matches_df, is_home=True)
    away_elo = get_team_elo(away_team, elo_df, matches_df, is_home=False)
    home_stats = get_team_recent_stats(home_team, matches_df)
    away_stats = get_team_recent_stats(away_team, matches_df)
    elo_diff = home_elo - away_elo
    elo_total = home_elo + away_elo

    # Get latest odds for this matchup if available
    odds = {'OddHome': None, 'OddDraw': None, 'OddAway': None}
    if matches_df is not None:
        matchup = matches_df[
            (matches_df['HomeTeam'] == home_team) & (matches_df['AwayTeam'] == away_team)
        ].sort_values('MatchDate')
        if len(matchup) > 0 and 'OddHome' in matchup.columns:
            last = matchup.iloc[-1]
            for k in odds:
                if k in last and pd.notna(last[k]):
                    odds[k] = last[k]
    # Compute implied probabilities
    impl_home = 1.0 / odds['OddHome'] if odds['OddHome'] else 0.33
    impl_draw = 1.0 / odds['OddDraw'] if odds['OddDraw'] else 0.33
    impl_away = 1.0 / odds['OddAway'] if odds['OddAway'] else 0.33
    margin = impl_home + impl_draw + impl_away
    impl_home /= margin
    impl_draw /= margin
    impl_away /= margin

    features = []
    for col in feature_columns:
        val = None
        if col == 'HomeElo':               val = home_elo
        elif col == 'AwayElo':             val = away_elo
        elif col == 'EloDifference':       val = elo_diff
        elif col == 'EloTotal':            val = elo_total
        elif col == 'Form3Home':           val = home_stats.get('Form3', 0)
        elif col == 'Form5Home':           val = home_stats.get('Form5', 0)
        elif col == 'Form3Away':           val = away_stats.get('Form3', 0)
        elif col == 'Form5Away':           val = away_stats.get('Form5', 0)
        elif col == 'Form3Diff':           val = home_stats.get('Form3', 0) - away_stats.get('Form3', 0)
        elif col == 'Form5Diff':           val = home_stats.get('Form5', 0) - away_stats.get('Form5', 0)
        elif col == 'HomeShots':           val = home_stats.get('Shots', 0)
        elif col == 'AwayShots':           val = away_stats.get('Shots', 0)
        elif col == 'ShotsDifference':     val = home_stats.get('Shots', 0) - away_stats.get('Shots', 0)
        elif col == 'HomeTarget':          val = home_stats.get('Target', 0)
        elif col == 'AwayTarget':          val = away_stats.get('Target', 0)
        elif col == 'HomeCorners':         val = home_stats.get('Corners', 0)
        elif col == 'AwayCorners':         val = away_stats.get('Corners', 0)
        elif col == 'CornersDifference':   val = home_stats.get('Corners', 0) - away_stats.get('Corners', 0)
        elif col == 'HomeFouls':           val = home_stats.get('Fouls', 0)
        elif col == 'AwayFouls':           val = away_stats.get('Fouls', 0)
        elif col == 'HomeYellow':          val = home_stats.get('Yellow', 0)
        elif col == 'AwayYellow':          val = away_stats.get('Yellow', 0)
        elif col == 'HomeRed':             val = home_stats.get('Red', 0)
        elif col == 'AwayRed':             val = away_stats.get('Red', 0)
        elif col == 'CardPointsHome':      val = home_stats.get('Yellow', 0) + 2 * home_stats.get('Red', 0)
        elif col == 'CardPointsAway':      val = away_stats.get('Yellow', 0) + 2 * away_stats.get('Red', 0)
        elif col == 'Year':                val = pd.Timestamp.now().year
        # Betting odds features
        elif col == 'ImpliedProbHome':     val = impl_home
        elif col == 'ImpliedProbDraw':     val = impl_draw
        elif col == 'ImpliedProbAway':     val = impl_away
        elif col == 'OddsDiffHomeAway':    val = impl_home - impl_away
        # Shot conversion
        elif col == 'HomeShotConversion':  val = home_stats.get('ShotConversion', 0)
        elif col == 'AwayShotConversion':  val = away_stats.get('ShotConversion', 0)
        elif col == 'ShotConversionDiff':  val = home_stats.get('ShotConversion', 0) - away_stats.get('ShotConversion', 0)
        # Win streaks
        elif col == 'Home_WinStreak':      val = home_stats.get('WinStreak', 0)
        elif col == 'Away_WinStreak':      val = away_stats.get('WinStreak', 0)
        elif col == 'WinStreakDiff':       val = home_stats.get('WinStreak', 0) - away_stats.get('WinStreak', 0)
        # Rolling stats (use recent stats as proxy)
        elif 'GoalsScored' in col and 'Home' in col:    val = home_stats.get('GoalsScored', 0)
        elif 'GoalsScored' in col and 'Away' in col:    val = away_stats.get('GoalsScored', 0)
        elif 'GoalsConceded' in col and 'Home' in col:  val = home_stats.get('GoalsConceded', 0)
        elif 'GoalsConceded' in col and 'Away' in col:  val = away_stats.get('GoalsConceded', 0)
        elif 'Points' in col and 'Home' in col:         val = home_stats.get('Points', 0)
        elif 'Points' in col and 'Away' in col:         val = away_stats.get('Points', 0)
        elif 'GoalDiff' in col and 'Home' in col:       val = home_stats.get('GoalDiff', 0)
        elif 'GoalDiff' in col and 'Away' in col:       val = away_stats.get('GoalDiff', 0)
        elif 'Won' in col and 'Home' in col:            val = home_stats.get('Points', 0) / 3 if home_stats.get('Points', 0) else 0
        elif 'Won' in col and 'Away' in col:            val = away_stats.get('Points', 0) / 3 if away_stats.get('Points', 0) else 0
        elif 'ShotsOnTarget' in col and 'Home' in col:  val = home_stats.get('Target', 0)
        elif 'ShotsOnTarget' in col and 'Away' in col:  val = away_stats.get('Target', 0)
        elif 'GoalsScored_Diff' in col:                 val = home_stats.get('GoalsScored', 0) - away_stats.get('GoalsScored', 0)
        else:                              val = 0
        features.append(val if val is not None else 0)
    return np.array([features]), home_elo, away_elo


# ──────────────────────────────────────────────
# 🚀 ENDPOINTS
# ──────────────────────────────────────────────

@router.post("/", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """
    Face o predicție pentru un meci de fotbal.
    """
    try:
        logger.info("Prediction request: %s vs %s", request.home_team, request.away_team)
        print(f"Prediction request: {request.home_team} vs {request.away_team}")
        rf_model, lr_model, xgb_model, le, feature_columns = load_models()

        if rf_model is None or feature_columns is None:
            logger.warning("Models not loaded - returning 503")
            print("Models not loaded - returning 503")
            raise HTTPException(
                status_code=503,
                detail="Modelele nu sunt încărcate! Rulează /train mai întâi."
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
        features, home_elo, away_elo = build_feature_vector(
            request.home_team, request.away_team, matches_df, elo_df, feature_columns
        )

        logger.info("Running prediction (HomeElo=%.1f, AwayElo=%.1f)", home_elo, away_elo)
        print(f"Running prediction (HomeElo={home_elo:.1f}, AwayElo={away_elo:.1f})")

        # Get probabilities from all 3 models
        rf_proba = rf_model.predict_proba(features)[0]    # classes: [-1, 0, 1]
        lr_proba = lr_model.predict_proba(features)[0]    # classes: [-1, 0, 1]
        xgb_proba_raw = xgb_model.predict_proba(features)[0]  # classes: encoded

        # Reorder XGB proba to match [-1, 0, 1] order
        xgb_class_order = le.inverse_transform(xgb_model.classes_)
        target_order = sorted(rf_model.classes_)
        xgb_col_map = [list(xgb_class_order).index(c) for c in target_order]
        xgb_proba = xgb_proba_raw[xgb_col_map]

        # Ensemble: weighted average of probabilities
        ensemble_proba = 0.4 * xgb_proba + 0.35 * rf_proba + 0.25 * lr_proba

        # Map to original labels: target_order = [-1, 0, 1]
        prediction = target_order[np.argmax(ensemble_proba)]

        prob_dict = {}
        for i, label in enumerate(target_order):
            prob_dict[str(int(label))] = ensemble_proba[i] * 100

        result_map = {
            1.0:  "🏠 Câștigă ACASĂ",
            0.0:  "🤝 EGAL",
            -1.0: "🚗 Câștigă DEPLASARE"
        }

        home_win_prob = prob_dict.get("1", 0)
        draw_prob = prob_dict.get("0", 0)
        away_win_prob = prob_dict.get("-1", 0)

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
            "away_win_prob": round(away_win_prob, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Eroare la predicție: {str(e)}"
        )
