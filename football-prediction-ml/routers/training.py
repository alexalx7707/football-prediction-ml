"""
Router for model training endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import os
import logging
import pandas as pd
import numpy as np
import joblib
import kagglehub
import optuna

logger = logging.getLogger("football-api.training")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/train", tags=["Training"])

# Global state
training_in_progress = False
training_status = {"status": "idle", "message": ""}


def compute_h2h_features(df):
    """Compute head-to-head features per row using only matches before this one
    between the same team pair. Stats are relative to the current row's home team
    (e.g. H2H_HomeWinRate = past wins by the current home team across the pair,
    regardless of which side they played on then). Assumes df is sorted by MatchDate.
    """
    home_teams = df['HomeTeam'].astype(str).values
    away_teams = df['AwayTeam'].astype(str).values
    ft_home = df['FTHome'].values
    ft_away = df['FTAway'].values
    ft_result = df['FTResult'].values

    n = len(df)
    h2h_home_winrate = np.zeros(n)
    h2h_draw_rate = np.zeros(n)
    h2h_total = np.zeros(n)
    h2h_avg_goaldiff = np.zeros(n)

    pair_history = {}

    for i in range(n):
        a, b = home_teams[i], away_teams[i]
        pk = (a, b) if a < b else (b, a)
        history = pair_history.get(pk)
        if history:
            wins = 0
            draws = 0
            gd_sum = 0.0
            count = len(history)
            for past_home, past_fhome, past_faway, past_result in history:
                if past_result == 'D':
                    draws += 1
                else:
                    if past_home == a:
                        if past_result == 'H':
                            wins += 1
                        gd_sum += past_fhome - past_faway
                    else:
                        if past_result == 'A':
                            wins += 1
                        gd_sum += past_faway - past_fhome
            h2h_home_winrate[i] = wins / count
            h2h_draw_rate[i] = draws / count
            h2h_total[i] = count
            h2h_avg_goaldiff[i] = gd_sum / count
        pair_history.setdefault(pk, []).append((a, ft_home[i], ft_away[i], ft_result[i]))

    return h2h_home_winrate, h2h_draw_rate, h2h_total, h2h_avg_goaldiff


def compute_side_form(df, team_col, points_series, goaldiff_series, prefix, windows=(5, 10)):
    """Rolling form computed only over matches where the team played on this side
    (e.g. home team's previous home matches). Returns a DataFrame indexed by the
    original df.index with new feature columns; NaN filled with 0.
    """
    side = pd.DataFrame({
        '_orig_idx': df.index,
        'MatchDate': df['MatchDate'].values,
        'Team': df[team_col].values,
        'Points': points_series.values,
        'GoalDiff': goaldiff_series.values,
    })
    side = side.sort_values(['Team', 'MatchDate']).reset_index(drop=True)

    new_cols = []
    for window in windows:
        for stat in ['Points', 'GoalDiff']:
            col = f'{prefix}_{stat}_Last{window}'
            side[col] = (
                side.groupby('Team')[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            new_cols.append(col)

    return side.set_index('_orig_idx')[new_cols].fillna(0)


def tune_random_forest(X_tune, y_tune, X_val, y_val, n_trials):
    """Optuna search over RF hyperparameters using a single chronological train/val
    holdout (much faster than CV; equivalent ranking quality for hyperparam selection).
    Returns (best_params, best_value)."""
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 400),
            'max_depth':         trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            'class_weight':      'balanced',
            'random_state':      42,
            'n_jobs':            -1,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tune, y_tune)
        proba = model.predict_proba(X_val)
        return -log_loss(y_val, proba, labels=model.classes_)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


def tune_logistic_regression(X_tune, y_tune, X_val, y_val, n_trials):
    """Optuna search for LR (Pipeline with StandardScaler). saga solver supports both
    L1 and L2; scaling is required for saga to converge well on mixed-scale features.
    Single train/val holdout; returns (best_params, best_value) with keys 'C', 'penalty'."""
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                C=C, penalty=penalty, solver='saga', max_iter=2000,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )),
        ])
        pipeline.fit(X_tune, y_tune)
        proba = pipeline.predict_proba(X_val)
        return -log_loss(y_val, proba, labels=pipeline.classes_)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


# ──────────────────────────────────────────────
# 📊 PYDANTIC MODELS
# ──────────────────────────────────────────────

class TrainingRequest(BaseModel):
    test_season: str = Field(default="2024/25", description="Sezonul pentru test")
    n_trials: int = Field(default=50, ge=1, le=200, description="Numărul de trial-uri Optuna pentru XGBoost (RF folosește n_trials // 2, LR până la 30)")
    tune_all_models: bool = Field(default=True, description="Dacă false, RF și LR folosesc parametri default (rapid). Dacă true, toate trei modelele sunt tunate cu Optuna.")

    class Config:
        json_schema_extra = {
            "example": {
                "test_season": "2024/25",
                "n_trials": 50,
                "tune_all_models": True
            }
        }


class TrainingResponse(BaseModel):
    message: str
    status: str
    accuracy_rf: float
    accuracy_lr: float
    accuracy_xgb: float
    accuracy_ensemble: float
    test_matches: int
    models_saved: bool


# ──────────────────────────────────────────────
# 🚀 ENDPOINTS
# ──────────────────────────────────────────────

@router.post("/", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Antrenează modelele cu date noi.
    """
    global training_in_progress, training_status

    if training_in_progress:
        raise HTTPException(
            status_code=409,
            detail="Antrenamentul este deja în progres! Așteaptă să se termine."
        )

    logger.info("Training started: test_season=%s, n_trials=%d", request.test_season, request.n_trials)
    print(f"Training started: test_season={request.test_season}, n_trials={request.n_trials}")
    training_in_progress = True
    training_status = {"status": "starting", "message": "Descent datelor..."}

    def train_background():
        global training_in_progress, training_status

        try:
            logger.info("Downloading dataset from Kaggle...")
            print("Downloading dataset from Kaggle...")
            training_status = {"status": "downloading", "message": "Descarcă date de pe Kaggle..."}

            path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
            matches_path = os.path.join(path, 'Matches.csv')
            df = pd.read_csv(matches_path)
            logger.info("Dataset downloaded: %d rows", len(df))
            print(f"Dataset downloaded: {len(df)} rows")

            training_status = {"status": "preprocessing", "message": "Preprocesare date..."}
            logger.info("Preprocessing data...")
            print("Preprocessing data...")

            df_model = df.copy()
            df_model['MatchDate'] = pd.to_datetime(df_model['MatchDate'], errors='coerce')
            df_model = df_model.dropna(subset=['MatchDate'])
            df_model = df_model.sort_values('MatchDate').reset_index(drop=True)

            def get_season(date):
                if pd.isna(date):
                    return np.nan
                if date.month >= 7:
                    return f"{date.year}/{str(date.year + 1)[-2:]}"
                else:
                    return f"{date.year - 1}/{str(date.year)[-2:]}"

            df_model['Season'] = df_model['MatchDate'].apply(get_season)
            df_model['Year'] = df_model['MatchDate'].dt.year
            df_model = df_model.dropna(subset=['Season'])
            df_model = df_model.dropna(subset=['HomeTeam', 'AwayTeam', 'FTResult'])

            def map_result(result):
                if result == 'H':   return 1
                elif result == 'D': return 0
                elif result == 'A': return -1
                else: return np.nan

            df_model['Result'] = df_model['FTResult'].apply(map_result)
            df_model = df_model.dropna(subset=['Result'])

            # Feature engineering
            logger.info("Feature engineering...")
            print("Feature engineering...")
            training_status = {"status": "features", "message": "Inginerie de caracteristici..."}

            feature_columns = ['HomeElo', 'AwayElo']
            form_features = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
            attacking_features = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget', 'HomeCorners', 'AwayCorners']
            discipline_features = ['HomeFouls', 'AwayFouls', 'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed']

            for feat in form_features + attacking_features + discipline_features:
                if feat in df_model.columns:
                    feature_columns.append(feat)

            if 'HomeElo' in df_model.columns and 'AwayElo' in df_model.columns:
                df_model['EloDifference'] = df_model['HomeElo'] - df_model['AwayElo']
                df_model['EloTotal'] = df_model['HomeElo'] + df_model['AwayElo']
                feature_columns.extend(['EloDifference', 'EloTotal'])

            if 'Form3Home' in df_model.columns and 'Form3Away' in df_model.columns:
                df_model['Form3Diff'] = df_model['Form3Home'] - df_model['Form3Away']
                feature_columns.append('Form3Diff')

            if 'Form5Home' in df_model.columns and 'Form5Away' in df_model.columns:
                df_model['Form5Diff'] = df_model['Form5Home'] - df_model['Form5Away']
                feature_columns.append('Form5Diff')

            if 'HomeShots' in df_model.columns and 'AwayShots' in df_model.columns:
                df_model['ShotsDifference'] = df_model['HomeShots'] - df_model['AwayShots']
                feature_columns.append('ShotsDifference')

            if 'HomeCorners' in df_model.columns and 'AwayCorners' in df_model.columns:
                df_model['CornersDifference'] = df_model['HomeCorners'] - df_model['AwayCorners']
                feature_columns.append('CornersDifference')

            # --- Betting odds features (strongest predictors — implied probabilities) ---
            # Prefer max-line odds (lower bookmaker margin), fall back to standard odds per-row.
            if all(c in df_model.columns for c in ['OddHome', 'OddDraw', 'OddAway']):
                if all(c in df_model.columns for c in ['MaxHome', 'MaxDraw', 'MaxAway']):
                    odd_home = df_model['MaxHome'].combine_first(df_model['OddHome'])
                    odd_draw = df_model['MaxDraw'].combine_first(df_model['OddDraw'])
                    odd_away = df_model['MaxAway'].combine_first(df_model['OddAway'])
                else:
                    odd_home = df_model['OddHome']
                    odd_draw = df_model['OddDraw']
                    odd_away = df_model['OddAway']
                df_model['ImpliedProbHome'] = 1.0 / odd_home
                df_model['ImpliedProbDraw'] = 1.0 / odd_draw
                df_model['ImpliedProbAway'] = 1.0 / odd_away
                margin = df_model['ImpliedProbHome'] + df_model['ImpliedProbDraw'] + df_model['ImpliedProbAway']
                df_model['ImpliedProbHome'] = df_model['ImpliedProbHome'] / margin
                df_model['ImpliedProbDraw'] = df_model['ImpliedProbDraw'] / margin
                df_model['ImpliedProbAway'] = df_model['ImpliedProbAway'] / margin
                df_model['OddsDiffHomeAway'] = df_model['ImpliedProbHome'] - df_model['ImpliedProbAway']
                feature_columns.extend(['ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway', 'OddsDiffHomeAway'])

            # --- Over/Under 2.5 goals market (orthogonal to 1X2) ---
            if all(c in df_model.columns for c in ['Over25', 'Under25']):
                if all(c in df_model.columns for c in ['MaxOver25', 'MaxUnder25']):
                    over25 = df_model['MaxOver25'].combine_first(df_model['Over25'])
                    under25 = df_model['MaxUnder25'].combine_first(df_model['Under25'])
                else:
                    over25 = df_model['Over25']
                    under25 = df_model['Under25']
                df_model['ImpliedProbOver25'] = 1.0 / over25
                df_model['ImpliedProbUnder25'] = 1.0 / under25
                ou_margin = df_model['ImpliedProbOver25'] + df_model['ImpliedProbUnder25']
                df_model['ImpliedProbOver25'] = (df_model['ImpliedProbOver25'] / ou_margin).fillna(0.5)
                df_model['ImpliedProbUnder25'] = (df_model['ImpliedProbUnder25'] / ou_margin).fillna(0.5)
                df_model['Over25Diff'] = (df_model['ImpliedProbOver25'] - df_model['ImpliedProbUnder25']).fillna(0)
                feature_columns.extend(['ImpliedProbOver25', 'ImpliedProbUnder25', 'Over25Diff'])

            # --- Asian handicap (efficiently priced, captures expected goal diff) ---
            if all(c in df_model.columns for c in ['HandiHome', 'HandiAway', 'HandiSize']):
                df_model['HandiSizeNum'] = df_model['HandiSize'].fillna(0)
                hi_home = 1.0 / df_model['HandiHome']
                hi_away = 1.0 / df_model['HandiAway']
                hi_margin = hi_home + hi_away
                df_model['HandiImpliedHome'] = (hi_home / hi_margin).fillna(0.5)
                df_model['HandiImpliedAway'] = (hi_away / hi_margin).fillna(0.5)
                feature_columns.extend(['HandiSizeNum', 'HandiImpliedHome', 'HandiImpliedAway'])

            # --- League / division as one-hot (different leagues, different dynamics) ---
            if 'Division' in df_model.columns:
                division_dummies = pd.get_dummies(df_model['Division'], prefix='Div', dummy_na=False).astype(int)
                df_model = pd.concat([df_model, division_dummies], axis=1)
                feature_columns.extend(division_dummies.columns.tolist())

            # --- Shot conversion rate (efficiency) ---
            if 'HomeShots' in df_model.columns and 'HomeTarget' in df_model.columns:
                df_model['HomeShotConversion'] = (df_model['HomeTarget'] / df_model['HomeShots'].replace(0, np.nan)).fillna(0)
                df_model['AwayShotConversion'] = (df_model['AwayTarget'] / df_model['AwayShots'].replace(0, np.nan)).fillna(0)
                df_model['ShotConversionDiff'] = df_model['HomeShotConversion'] - df_model['AwayShotConversion']
                feature_columns.extend(['HomeShotConversion', 'AwayShotConversion', 'ShotConversionDiff'])

            # --- Goal difference (FTHome - FTAway as a feature for rolling) ---
            df_model['GoalDifference'] = df_model['FTHome'] - df_model['FTAway']

            for col in feature_columns:
                if col in df_model.columns:
                    df_model[col] = df_model[col].fillna(0)

            # Head-to-head features (history between the same team pair)
            logger.info("Calculating head-to-head features...")
            print("Calculating head-to-head features...")
            training_status = {"status": "h2h", "message": "Calculez statistici head-to-head..."}
            h2h_hw, h2h_dr, h2h_tot, h2h_gd = compute_h2h_features(df_model)
            df_model['H2H_HomeWinRate'] = h2h_hw
            df_model['H2H_DrawRate'] = h2h_dr
            df_model['H2H_TotalMatches'] = h2h_tot
            df_model['H2H_AvgGoalDiff'] = h2h_gd
            feature_columns.extend(['H2H_HomeWinRate', 'H2H_DrawRate', 'H2H_TotalMatches', 'H2H_AvgGoalDiff'])

            # Rolling features (computed across ALL of a team's matches, home + away)
            logger.info("Calculating rolling features...")
            print("Calculating rolling features...")
            training_status = {"status": "rolling", "message": "Calculez rolling features..."}

            # Build unified match records: each match creates two rows (one per team)
            is_draw_series = (df_model['FTResult'] == 'D').astype(int)
            home_records = pd.DataFrame({
                'MatchDate': df_model['MatchDate'],
                'Team': df_model['HomeTeam'],
                'GoalsScored': df_model['FTHome'],
                'GoalsConceded': df_model['FTAway'],
                'Points': df_model['FTResult'].map({'H': 3, 'D': 1, 'A': 0}),
                'Won': (df_model['FTResult'] == 'H').astype(int),
                'IsDraw': is_draw_series,
                'GoalDiff': df_model['FTHome'] - df_model['FTAway'],
                'ShotsOnTarget': df_model.get('HomeTarget', pd.Series(0, index=df_model.index)),
                'OrigIdx': df_model.index,
                'Role': 'home'
            })
            away_records = pd.DataFrame({
                'MatchDate': df_model['MatchDate'],
                'Team': df_model['AwayTeam'],
                'GoalsScored': df_model['FTAway'],
                'GoalsConceded': df_model['FTHome'],
                'Points': df_model['FTResult'].map({'H': 0, 'D': 1, 'A': 3}),
                'Won': (df_model['FTResult'] == 'A').astype(int),
                'IsDraw': is_draw_series,
                'GoalDiff': df_model['FTAway'] - df_model['FTHome'],
                'ShotsOnTarget': df_model.get('AwayTarget', pd.Series(0, index=df_model.index)),
                'OrigIdx': df_model.index,
                'Role': 'away'
            })

            all_team_matches = pd.concat([home_records, away_records], ignore_index=True)
            all_team_matches = all_team_matches.sort_values(['Team', 'MatchDate']).reset_index(drop=True)

            # Days-since-last-match per team (rest / fixture congestion)
            all_team_matches['DaysSinceLast'] = (
                all_team_matches.groupby('Team')['MatchDate']
                .diff().dt.days.fillna(14)
            )

            # Compute rolling averages per team across all their matches
            for window in [5, 10]:
                for stat in ['GoalsScored', 'GoalsConceded', 'Points', 'GoalDiff', 'Won', 'ShotsOnTarget', 'IsDraw']:
                    all_team_matches[f'{stat}_Last{window}'] = (
                        all_team_matches.groupby('Team')[stat]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    )

            # Win streak: count consecutive wins looking backwards
            def calc_streak(series):
                streak = pd.Series(0, index=series.index, dtype=int)
                current = 0
                for i in range(len(series)):
                    if i == 0:
                        streak.iloc[i] = 0  # no history yet
                    else:
                        if series.iloc[i - 1] == 1:
                            current += 1
                        else:
                            current = 0
                        streak.iloc[i] = current
                return streak

            all_team_matches['WinStreak'] = (
                all_team_matches.groupby('Team')['Won']
                .transform(calc_streak)
            )

            # Split back and merge to df_model with Home_/Away_ prefixes
            roll_cols = [c for c in all_team_matches.columns
                         if 'Last5' in c or 'Last10' in c or c in ('WinStreak', 'DaysSinceLast')]

            home_rolling = (all_team_matches[all_team_matches['Role'] == 'home']
                           [['OrigIdx'] + roll_cols].set_index('OrigIdx'))
            home_rolling.columns = [f'Home_{c}' for c in home_rolling.columns]

            away_rolling = (all_team_matches[all_team_matches['Role'] == 'away']
                           [['OrigIdx'] + roll_cols].set_index('OrigIdx'))
            away_rolling.columns = [f'Away_{c}' for c in away_rolling.columns]

            df_model = df_model.join(home_rolling).join(away_rolling)

            for window in [5, 10]:
                df_model[f'GoalsScored_Diff_Last{window}'] = (
                    df_model[f'Home_GoalsScored_Last{window}'] - df_model[f'Away_GoalsScored_Last{window}']
                )

            rolling_feature_cols = [col for col in df_model.columns
                                    if 'Last5' in col or 'Last10' in col
                                    or 'WinStreak' in col or 'DaysSinceLast' in col]
            for col in rolling_feature_cols:
                df_model[col] = df_model[col].fillna(0)
                if col not in feature_columns:
                    feature_columns.append(col)

            # Streak difference
            if 'Home_WinStreak' in df_model.columns and 'Away_WinStreak' in df_model.columns:
                df_model['WinStreakDiff'] = df_model['Home_WinStreak'] - df_model['Away_WinStreak']
                feature_columns.append('WinStreakDiff')

            # Rest difference (positive = home team had more rest)
            if 'Home_DaysSinceLast' in df_model.columns and 'Away_DaysSinceLast' in df_model.columns:
                df_model['RestDiff'] = df_model['Home_DaysSinceLast'] - df_model['Away_DaysSinceLast']
                feature_columns.append('RestDiff')

            # Home-only / Away-only form (some teams are fortress at home / road disasters)
            home_at_home = compute_side_form(
                df_model, 'HomeTeam',
                df_model['FTResult'].map({'H': 3, 'D': 1, 'A': 0}),
                df_model['FTHome'] - df_model['FTAway'],
                'HomeForm_AtHome'
            )
            df_model = df_model.join(home_at_home)
            feature_columns.extend(home_at_home.columns.tolist())

            away_at_away = compute_side_form(
                df_model, 'AwayTeam',
                df_model['FTResult'].map({'H': 0, 'D': 1, 'A': 3}),
                df_model['FTAway'] - df_model['FTHome'],
                'AwayForm_AtAway'
            )
            df_model = df_model.join(away_at_away)
            feature_columns.extend(away_at_away.columns.tolist())

            # Train / Validation / Test split (Phase D)
            # Validation = season immediately before TEST_SEASON; used for hyperparam
            # tuning. After tuning, models are refit on (train + val) for final test eval.
            logger.info("Splitting train/val/test for season %s", request.test_season)
            print(f"Splitting train/val/test for season {request.test_season}")
            training_status = {"status": "splitting", "message": "Split train/val/test..."}

            TEST_SEASON = request.test_season

            def _prev_season(season):
                start = int(season.split('/')[0])
                return f"{start - 1}/{str(start)[-2:]}"

            VAL_SEASON = _prev_season(TEST_SEASON)

            test_df = df_model[df_model['Season'] == TEST_SEASON].copy()
            val_df = df_model[df_model['Season'] == VAL_SEASON].copy()
            train_df = df_model[~df_model['Season'].isin([TEST_SEASON, VAL_SEASON])].copy()

            if len(test_df) == 0:
                raise ValueError(f"Nu există meciuri pentru sezonul {TEST_SEASON}")

            if len(val_df) == 0:
                # Fallback: take the most recent ~15% of train chronologically as val
                logger.info("Validation season %s empty; using last 15%% of train as val", VAL_SEASON)
                print(f"Validation season {VAL_SEASON} empty; using last 15% of train as val")
                train_df = train_df.sort_values('MatchDate').reset_index(drop=True)
                cutoff = int(len(train_df) * 0.85)
                val_df = train_df.iloc[cutoff:].copy()
                train_df = train_df.iloc[:cutoff].copy()

            X_train = train_df[feature_columns].fillna(0)
            y_train = train_df['Result']
            X_val = val_df[feature_columns].fillna(0)
            y_val = val_df['Result']
            X_test = test_df[feature_columns].fillna(0)
            y_test = test_df['Result']
            logger.info("Train: %d, Val: %d, Test: %d, Features: %d",
                        len(X_train), len(X_val), len(X_test), len(feature_columns))
            print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {len(feature_columns)}")

            # Tuning subsample: hyperparameter rankings are stable on a ~60k subset.
            # Take the most-recent training rows (most relevant for the test period).
            TUNE_MAX_ROWS = 60000
            if len(X_train) > TUNE_MAX_ROWS:
                tune_idx = train_df.sort_values('MatchDate').tail(TUNE_MAX_ROWS).index
                X_tune = X_train.loc[tune_idx]
                y_tune = y_train.loc[tune_idx]
            else:
                X_tune = X_train
                y_tune = y_train
            logger.info("Tuning subsample: %d rows", len(X_tune))
            print(f"Tuning subsample: {len(X_tune)} rows")

            # Combined train+val frame for final refits after tuning
            X_full_train = pd.concat([X_train, X_val])
            y_full_train = pd.concat([y_train, y_val])

            # Training
            logger.info("Training models...")
            print("Training models...")
            training_status = {"status": "training", "message": "Antrenez modelele..."}

            # --- Label encoding for XGBoost (fit on full train+val so it sees all classes) ---
            le = LabelEncoder()
            le.fit(y_full_train)
            y_tune_enc = le.transform(y_tune)
            y_val_enc = le.transform(y_val)
            y_full_train_enc = le.transform(y_full_train)

            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights_tune = compute_sample_weight('balanced', y_tune_enc)
            sample_weights_full = compute_sample_weight('balanced', y_full_train_enc)

            # --- RandomForest: Optuna on subsample / val, then refit on train+val ---
            if request.tune_all_models:
                rf_trials = max(10, request.n_trials // 2)
                logger.info("Optuna RF search starting (%d trials)...", rf_trials)
                print(f"Optuna RF search starting ({rf_trials} trials)...")
                training_status = {"status": "tuning_rf", "message": f"Optuna RF caută parametri ({rf_trials} trial-uri)..."}
                rf_best_params, rf_best_score = tune_random_forest(X_tune, y_tune, X_val, y_val, rf_trials)
                logger.info("RF best params: %s (val neg_log_loss=%.4f)", rf_best_params, rf_best_score)
                print(f"RF best params: {rf_best_params}, val neg_log_loss={rf_best_score:.4f}")
                rf_model = RandomForestClassifier(
                    **rf_best_params, class_weight='balanced', random_state=42, n_jobs=-1
                )
            else:
                rf_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
                )
            logger.info("Refitting RF on full train+val (%d rows)...", len(X_full_train))
            print(f"Refitting RF on full train+val ({len(X_full_train)} rows)...")
            rf_model.fit(X_full_train, y_full_train)

            # --- LogisticRegression: Optuna on subsample / val, then refit on train+val ---
            if request.tune_all_models:
                lr_trials = max(10, min(30, request.n_trials))
                logger.info("Optuna LR search starting (%d trials)...", lr_trials)
                print(f"Optuna LR search starting ({lr_trials} trials)...")
                training_status = {"status": "tuning_lr", "message": f"Optuna LR caută parametri ({lr_trials} trial-uri)..."}
                lr_best_params, lr_best_score = tune_logistic_regression(X_tune, y_tune, X_val, y_val, lr_trials)
                logger.info("LR best params: %s (val neg_log_loss=%.4f)", lr_best_params, lr_best_score)
                print(f"LR best params: {lr_best_params}, val neg_log_loss={lr_best_score:.4f}")
                lr_inner = LogisticRegression(
                    **lr_best_params, solver='saga', max_iter=2000,
                    class_weight='balanced', random_state=42, n_jobs=-1,
                )
            else:
                lr_inner = LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced'
                )
            lr_model = Pipeline([('scaler', StandardScaler()), ('lr', lr_inner)])
            logger.info("Refitting LR on full train+val (%d rows)...", len(X_full_train))
            print(f"Refitting LR on full train+val ({len(X_full_train)} rows)...")
            lr_model.fit(X_full_train, y_full_train)

            # --- XGBoost: Optuna on subsample / val, then refit on train+val ---
            logger.info("Optuna XGB search starting (%d trials)...", request.n_trials)
            print(f"Optuna XGB search starting ({request.n_trials} trials)...")
            training_status = {"status": "tuning_xgb", "message": f"Optuna XGB caută parametri ({request.n_trials} trial-uri)..."}

            xgb_class_count = len(le.classes_)

            def objective(trial):
                params = {
                    'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
                    'max_depth':         trial.suggest_int('max_depth', 3, 8),
                    'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
                    'eval_metric':       'mlogloss',
                    'random_state':      42,
                    'n_jobs':            -1,
                }
                model = XGBClassifier(**params)
                model.fit(X_tune, y_tune_enc, sample_weight=sample_weights_tune)
                proba = model.predict_proba(X_val)
                return -log_loss(y_val_enc, proba, labels=list(range(xgb_class_count)))

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=request.n_trials, show_progress_bar=True)

            logger.info("XGB best params: %s (val neg_log_loss=%.4f)", study.best_params, study.best_value)
            print(f"XGB best params: {study.best_params}")
            print(f"XGB val neg_log_loss: {study.best_value:.4f}")

            logger.info("Refitting XGB on full train+val (%d rows)...", len(X_full_train))
            print(f"Refitting XGB on full train+val ({len(X_full_train)} rows)...")
            xgb_model = XGBClassifier(**study.best_params, eval_metric='mlogloss', random_state=42, n_jobs=-1)
            xgb_model.fit(X_full_train, y_full_train_enc, sample_weight=sample_weights_full)

            # `y_train_enc` retained name for downstream evaluation code that referenced it
            y_train_enc = y_full_train_enc

            # Evaluation
            logger.info("Evaluating models...")
            print("Evaluating models...")
            training_status = {"status": "evaluation", "message": "Evaluez modelele..."}

            y_pred_rf = rf_model.predict(X_test)
            y_pred_lr = lr_model.predict(X_test)
            y_pred_xgb_enc = xgb_model.predict(X_test)
            y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)

            # Ensemble: weighted average of probabilities from all 3 models
            # RF and LR predict original labels (-1, 0, 1); XGB uses encoded labels
            rf_proba = rf_model.predict_proba(X_test)   # classes: [-1, 0, 1]
            lr_proba = lr_model.predict_proba(X_test)   # classes: [-1, 0, 1]
            xgb_proba = xgb_model.predict_proba(X_test) # classes: encoded
            # Reorder XGB proba columns to match [-1, 0, 1] order
            xgb_class_order = le.inverse_transform(xgb_model.classes_)
            target_order = sorted(rf_model.classes_)
            xgb_col_map = [list(xgb_class_order).index(c) for c in target_order]
            xgb_proba_reordered = xgb_proba[:, xgb_col_map]

            ensemble_proba = 0.4 * xgb_proba_reordered + 0.35 * rf_proba + 0.25 * lr_proba
            ensemble_preds = np.array(target_order)[np.argmax(ensemble_proba, axis=1)]

            acc_rf = accuracy_score(y_test, y_pred_rf)
            acc_lr = accuracy_score(y_test, y_pred_lr)
            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            acc_ensemble = accuracy_score(y_test, ensemble_preds)
            logger.info("Accuracy RF=%.4f, LR=%.4f, XGB=%.4f, Ensemble=%.4f", acc_rf, acc_lr, acc_xgb, acc_ensemble)
            print(f"Accuracy RF={acc_rf:.4f}, LR={acc_lr:.4f}, XGB={acc_xgb:.4f}, Ensemble={acc_ensemble:.4f}")

            # Save
            logger.info("Saving models...")
            print("Saving models...")
            training_status = {"status": "saving", "message": "Salvez modelele..."}

            os.makedirs('models', exist_ok=True)
            joblib.dump(rf_model, 'models/random_forest_model.pkl')
            joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
            joblib.dump(xgb_model, 'models/xgb_model.pkl')
            joblib.dump(le, 'models/label_encoder.pkl')
            joblib.dump(feature_columns, 'models/feature_columns.pkl')

            logger.info("Training completed successfully!")
            print("Training completed successfully!")
            training_status = {
                "status": "completed",
                "message": "Antrenament completat cu succes!",
                "accuracy_rf": round(acc_rf * 100, 2),
                "accuracy_lr": round(acc_lr * 100, 2),
                "accuracy_xgb": round(acc_xgb * 100, 2),
                "accuracy_ensemble": round(acc_ensemble * 100, 2),
                "test_matches": len(y_test),
                "test_season": request.test_season
            }
        except Exception as e:
            logger.error("Training error: %s", str(e), exc_info=True)
            print(f"Training error: {str(e)}")
            training_status = {"status": "error", "message": f"Eroare: {str(e)}"}
        finally:
            training_in_progress = False

    background_tasks.add_task(train_background)

    return {
        "message": "Antrenamentul a început în background...",
        "status": "started",
        "accuracy_rf": 0.0,
        "accuracy_lr": 0.0,
        "accuracy_xgb": 0.0,
        "accuracy_ensemble": 0.0,
        "test_matches": 0,
        "models_saved": False
    }


@router.get("/status")
async def get_training_status():
    """Obține status-ul antrenamentului curent"""
    return training_status
