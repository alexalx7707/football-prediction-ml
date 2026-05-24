"""
Router for importing dataset data into MongoDB.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import math
import logging
import pandas as pd
import kagglehub

from db import get_db

logger = logging.getLogger("football-api.data")

router = APIRouter(prefix="/data", tags=["Data"])


# ──────────────────────────────────────────────
# 📊 PYDANTIC MODELS
# ──────────────────────────────────────────────

class ImportResponse(BaseModel):
    message: str
    matches_imported: int
    elo_ratings_imported: int


class CollectionStats(BaseModel):
    collection: str
    document_count: int


class DataStatsResponse(BaseModel):
    collections: list[CollectionStats]


class TeamsResponse(BaseModel):
    teams: list[str] = Field(..., description="Sorted distinct team names available for predictions")


class MatchQueryParams(BaseModel):
    home_team: Optional[str] = Field(None, description="Filter by home team")
    away_team: Optional[str] = Field(None, description="Filter by away team")
    season: Optional[str] = Field(None, description="Filter by season (e.g. 2024/25)")
    limit: int = Field(default=0, ge=0, description="Max results to return (0 = no limit)")


class EloQueryParams(BaseModel):
    club: Optional[str] = Field(None, description="Filter by club name")
    limit: int = Field(default=0, ge=0, description="Max results to return (0 = no limit)")


# ──────────────────────────────────────────────
# 🔧 HELPERS
# ──────────────────────────────────────────────

def _clean_record(record: dict) -> dict:
    """Replace NaN/inf values with None for MongoDB compatibility."""
    cleaned = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


# ──────────────────────────────────────────────
# 🚀 ENDPOINTS
# ──────────────────────────────────────────────

@router.post("/import", response_model=ImportResponse)
async def import_dataset():
    """
    Downloads the Kaggle dataset and imports Matches and EloRatings into MongoDB.
    Existing data is replaced (collections are dropped first).
    """
    logger.info("Starting dataset import into MongoDB...")
    print("Starting dataset import into MongoDB...")
    try:
        path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
        logger.info("Dataset downloaded to: %s", path)
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        logger.error("Failed to download dataset: %s", str(e))
        print(f"Failed to download dataset: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to download dataset: {e}")

    db = get_db()

    # --- Import Matches ---
    matches_path = os.path.join(path, "Matches.csv")
    if not os.path.exists(matches_path):
        raise HTTPException(status_code=404, detail="Matches.csv not found in dataset")

    matches_df = pd.read_csv(matches_path)
    matches_records = [_clean_record(r) for r in matches_df.to_dict(orient="records")]

    db.drop_collection("matches")
    matches_count = 0
    if matches_records:
        db.matches.insert_many(matches_records)
        matches_count = len(matches_records)
    logger.info("Imported %d matches into MongoDB", matches_count)
    print(f"Imported {matches_count} matches into MongoDB")

    # Create indexes for common queries
    db.matches.create_index("HomeTeam")
    db.matches.create_index("AwayTeam")
    db.matches.create_index("MatchDate")
    db.matches.create_index("Season")

    # --- Import EloRatings ---
    elo_path = os.path.join(path, "EloRatings.csv")
    elo_count = 0
    if os.path.exists(elo_path):
        elo_df = pd.read_csv(elo_path)
        elo_records = [_clean_record(r) for r in elo_df.to_dict(orient="records")]
        db.drop_collection("elo_ratings")
        if elo_records:
            db.elo_ratings.insert_many(elo_records)
            elo_count = len(elo_records)
        db.elo_ratings.create_index("club")
        db.elo_ratings.create_index("date")
    logger.info("Imported %d Elo ratings into MongoDB", elo_count)
    print(f"Imported {elo_count} Elo ratings into MongoDB")

    return ImportResponse(
        message="Dataset imported successfully into MongoDB",
        matches_imported=matches_count,
        elo_ratings_imported=elo_count,
    )


@router.get("/stats", response_model=DataStatsResponse)
async def data_stats():
    """Returns document counts for each collection in the database."""
    db = get_db()
    collections = db.list_collection_names()
    stats = []
    for name in sorted(collections):
        count = db[name].count_documents({})
        stats.append(CollectionStats(collection=name, document_count=count))
    return DataStatsResponse(collections=stats)


@router.get("/teams", response_model=TeamsResponse)
async def get_teams():
    """
    Returns the sorted list of distinct team names available for predictions —
    intended for populating a dropdown in the frontend.

    The names match exactly what POST /predict/ expects, because both ultimately come
    from the Kaggle Matches.csv. Reads from MongoDB first (fast; populated by
    POST /data/import), and falls back to downloading the Kaggle dataset if MongoDB
    holds no matches, so the dropdown stays usable even before an import.
    """
    teams = set()

    # 1) Prefer MongoDB — fast and consistent with the rest of /data/*
    try:
        db = get_db()
        for col in ("HomeTeam", "AwayTeam"):
            teams.update(db.matches.distinct(col))
    except Exception as e:
        logger.warning("Could not read teams from MongoDB: %s", str(e))
        print(f"Could not read teams from MongoDB: {str(e)}")

    # 2) Fall back to the Kaggle dataset if MongoDB is empty/unavailable
    if not teams:
        logger.info("MongoDB has no teams — deriving list from Kaggle dataset")
        print("MongoDB has no teams — deriving list from Kaggle dataset")
        try:
            path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
            matches_df = pd.read_csv(
                os.path.join(path, "Matches.csv"), usecols=["HomeTeam", "AwayTeam"]
            )
            teams.update(matches_df["HomeTeam"].dropna().tolist())
            teams.update(matches_df["AwayTeam"].dropna().tolist())
        except Exception as e:
            logger.error("Failed to load teams from Kaggle: %s", str(e))
            print(f"Failed to load teams from Kaggle: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Failed to load team list: {e}")

    # Drop NaN/empty, normalize to str, sort
    cleaned = sorted({
        str(t).strip() for t in teams
        if t is not None and not (isinstance(t, float) and math.isnan(t)) and str(t).strip()
    })
    logger.info("Returning %d teams", len(cleaned))
    print(f"Returning {len(cleaned)} teams")
    return TeamsResponse(teams=cleaned)


@router.post("/matches")
async def get_matches(params: MatchQueryParams):
    """
    Query matches stored in MongoDB with optional filters.
    """
    db = get_db()
    query = {}
    if params.home_team:
        query["HomeTeam"] = params.home_team
    if params.away_team:
        query["AwayTeam"] = params.away_team
    if params.season:
        query["Season"] = params.season

    logger.info("Querying matches: %s (limit=%d)", query, params.limit)
    print(f"Querying matches: {query} (limit={params.limit})")
    cursor = db.matches.find(query, {"_id": 0})
    if params.limit > 0:
        cursor = cursor.limit(params.limit)
    results = list(cursor)
    logger.info("Returned %d matches", len(results))
    print(f"Returned {len(results)} matches")
    return results


@router.post("/elo")
async def get_elo_ratings(params: EloQueryParams):
    """
    Query Elo ratings stored in MongoDB with optional club filter.
    """
    db = get_db()
    query = {}
    if params.club:
        query["club"] = params.club

    logger.info("Querying Elo ratings: %s (limit=%d)", query, params.limit)
    print(f"Querying Elo ratings: {query} (limit={params.limit})")
    cursor = db.elo_ratings.find(query, {"_id": 0}).sort("date", -1)
    if params.limit > 0:
        cursor = cursor.limit(params.limit)
    results = list(cursor)
    logger.info("Returned %d Elo ratings", len(results))
    print(f"Returned {len(results)} Elo ratings")
    return results
