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

logger = logging.getLogger("football-api.training")
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/train", tags=["Training"])

# Global state
training_in_progress = False
training_status = {"status": "idle", "message": ""}


# ──────────────────────────────────────────────
# 📊 PYDANTIC MODELS
# ──────────────────────────────────────────────

class TrainingRequest(BaseModel):
    test_season: str = Field(default="2024/25", description="Sezonul pentru test")
    n_estimators: int = Field(default=300, description="Numărul de estimatori XGBoost")

    class Config:
        json_schema_extra = {
            "example": {
                "test_season": "2024/25",
                "n_estimators": 300
            }
        }


class TrainingResponse(BaseModel):
    message: str
    status: str
    accuracy_rf: float
    accuracy_lr: float
    accuracy_xgb: float
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

    logger.info("Training started: test_season=%s, n_estimators=%d", request.test_season, request.n_estimators)
    print(f"Training started: test_season={request.test_season}, n_estimators={request.n_estimators}")
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

            if 'Year' in df_model.columns:
                feature_columns.append('Year')

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

            for col in feature_columns:
                if col in df_model.columns:
                    df_model[col] = df_model[col].fillna(0)

            # Rolling features
            logger.info("Calculating rolling features...")
            print("Calculating rolling features...")
            training_status = {"status": "rolling", "message": "Calculez rolling features..."}

            def match_points_home(result):
                if result == 'H': return 3
                elif result == 'D': return 1
                else: return 0

            df_model['PointsHome'] = df_model['FTResult'].apply(match_points_home)

            for window in [5, 10]:
                df_model[f'Home_GoalsScored_Last{window}'] = (
                    df_model.groupby('HomeTeam')['FTHome']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df_model[f'Away_GoalsScored_Last{window}'] = (
                    df_model.groupby('AwayTeam')['FTAway']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df_model[f'Home_GoalsConceded_Last{window}'] = (
                    df_model.groupby('HomeTeam')['FTAway']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df_model[f'Away_GoalsConceded_Last{window}'] = (
                    df_model.groupby('AwayTeam')['FTHome']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df_model[f'Home_Points_Last{window}'] = (
                    df_model.groupby('HomeTeam')['PointsHome']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                df_model[f'GoalsScored_Diff_Last{window}'] = (
                    df_model[f'Home_GoalsScored_Last{window}'] - df_model[f'Away_GoalsScored_Last{window}']
                )

            rolling_feature_cols = [col for col in df_model.columns if 'Last5' in col or 'Last10' in col]
            for col in rolling_feature_cols:
                df_model[col] = df_model[col].fillna(0)
                if col not in feature_columns:
                    feature_columns.append(col)

            # Train/Test split
            logger.info("Splitting train/test for season %s", request.test_season)
            print(f"Splitting train/test for season {request.test_season}")
            training_status = {"status": "splitting", "message": "Split train/test..."}

            TEST_SEASON = request.test_season
            train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
            test_df = df_model[df_model['Season'] == TEST_SEASON].copy()

            if len(test_df) == 0:
                raise ValueError(f"Nu există meciuri pentru sezonul {TEST_SEASON}")

            X_train = train_df[feature_columns].fillna(0)
            y_train = train_df['Result']
            X_test = test_df[feature_columns].fillna(0)
            y_test = test_df['Result']
            logger.info("Train: %d rows, Test: %d rows, Features: %d", len(X_train), len(X_test), len(feature_columns))
            print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows, Features: {len(feature_columns)}")

            # Training
            logger.info("Training models...")
            print("Training models...")
            training_status = {"status": "training", "message": "Antrenez modelele..."}

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)

            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train, y_train)

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)

            xgb_model = XGBClassifier(
                n_estimators=request.n_estimators,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=2
            )
            xgb_model.fit(X_train, y_train_enc)

            # Evaluation
            logger.info("Evaluating models...")
            print("Evaluating models...")
            training_status = {"status": "evaluation", "message": "Evaluez modelele..."}

            y_pred_rf = rf_model.predict(X_test)
            y_pred_xgb_enc = xgb_model.predict(X_test)
            y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)

            acc_rf = accuracy_score(y_test, y_pred_rf)
            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            logger.info("Accuracy RF=%.4f, XGB=%.4f", acc_rf, acc_xgb)
            print(f"Accuracy RF={acc_rf:.4f}, XGB={acc_xgb:.4f}")

            # Save
            logger.info("Saving models...")
            print("Saving models...")
            training_status = {"status": "saving", "message": "Salvez modelele..."}

            os.makedirs('models', exist_ok=True)
            joblib.dump(rf_model, 'models/random_forest_model.pkl')
            joblib.dump(xgb_model, 'models/xgb_model.pkl')
            joblib.dump(le, 'models/label_encoder.pkl')
            joblib.dump(feature_columns, 'models/feature_columns.pkl')

            logger.info("Training completed successfully!")
            print("Training completed successfully!")
            training_status = {"status": "completed", "message": "Antrenament completat cu succes!"}

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
        "test_matches": 0,
        "models_saved": False
    }


@router.get("/status")
async def get_training_status():
    """Obține status-ul antrenamentului curent"""
    return training_status
