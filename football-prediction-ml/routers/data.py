"""
Router for importing dataset data into MongoDB.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
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
    home_team: Optional[str] = Field(None, description="Filter by home team (case-insensitive)")
    away_team: Optional[str] = Field(None, description="Filter by away team (case-insensitive)")
    season: Optional[str] = Field(None, description="Filter by season, e.g. '2024/25' (also accepts '2024/2025')")
    date_from: Optional[str] = Field(None, description="Earliest match date, ISO YYYY-MM-DD (inclusive)")
    date_to: Optional[str] = Field(None, description="Latest match date, ISO YYYY-MM-DD (inclusive)")
    order: str = Field(default="desc", description="Sort by match date: 'desc' (newest first, default) or 'asc'")
    limit: int = Field(default=0, ge=0, description="Max results to return (0 = no limit). Sort is applied before the limit.")


class EloQueryParams(BaseModel):
    club: Optional[str] = Field(None, description="Filter by club name")
    limit: int = Field(default=0, ge=0, description="Max results to return (0 = no limit)")


class TeamSummaryParams(BaseModel):
    team: Optional[str] = Field(None, description="Team name (required, case-insensitive)")
    season: Optional[str] = Field(None, description="Season filter, e.g. '2024/25'. Use this OR a date window.")
    date_from: Optional[str] = Field(None, description="Earliest match date, ISO YYYY-MM-DD (inclusive)")
    date_to: Optional[str] = Field(None, description="Latest match date, ISO YYYY-MM-DD (inclusive)")


class HeadToHeadParams(BaseModel):
    team_a: Optional[str] = Field(None, description="First team (required, case-insensitive)")
    team_b: Optional[str] = Field(None, description="Second team (required, case-insensitive)")
    season: Optional[str] = Field(None, description="Season filter, e.g. '2024/25'. Use this OR a date window.")
    date_from: Optional[str] = Field(None, description="Earliest match date, ISO YYYY-MM-DD (inclusive)")
    date_to: Optional[str] = Field(None, description="Latest match date, ISO YYYY-MM-DD (inclusive)")
    limit: int = Field(default=10, ge=0, description="Max number of recent meetings to include in 'recent'")


class StandingsParams(BaseModel):
    season: Optional[str] = Field(None, description="Season (required), e.g. '2024/25'")
    league: Optional[str] = Field(None, description="League / Division code (e.g. 'E0', 'F1'). Required when the season spans multiple leagues.")


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


def _team_name_map(db) -> dict:
    """Lowercase-stripped name -> canonical stored name, from both home and away appearances."""
    names = set(db.matches.distinct("HomeTeam")) | set(db.matches.distinct("AwayTeam"))
    out = {}
    for n in names:
        if n is None:
            continue
        s = str(n).strip()
        if s:
            out[s.lower()] = s
    return out


def _resolve_team(db, name, name_map=None):
    """Resolve a possibly mis-cased / padded team name to its canonical stored form, or None."""
    if not name or not str(name).strip():
        return None
    if name_map is None:
        name_map = _team_name_map(db)
    return name_map.get(str(name).strip().lower())


def _season_bounds(season: str):
    """
    '2024/25' (or '2024/2025') -> ('2024-07-01', '2025-06-30').

    Mirrors training's get_season (routers/training.py): a season runs July 1 of the
    start year through June 30 of the next year. Lenient on the trailing-year format.
    Raises ValueError if the start year can't be parsed.
    """
    s = str(season).strip()
    try:
        start = int(s.split("/")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid season '{season}'. Expected format like '2024/25'.")
    return f"{start:04d}-07-01", f"{start + 1:04d}-06-30"


def _season_of(date_str: str):
    """ISO 'YYYY-MM-DD' -> canonical season string '2024/25' (July cutoff), or None."""
    try:
        y = int(date_str[:4])
        m = int(date_str[5:7])
    except (ValueError, TypeError):
        return None
    if m >= 7:
        return f"{y}/{str(y + 1)[-2:]}"
    return f"{y - 1}/{str(y)[-2:]}"


def _date_range(season=None, date_from=None, date_to=None):
    """
    Resolve a (lo, hi) inclusive ISO date window from a season and/or explicit dates.
    Explicit date_from/date_to override the season bounds. Raises ValueError on bad season.
    """
    lo = hi = None
    if season:
        lo, hi = _season_bounds(season)
    if date_from:
        lo = str(date_from).strip()
    if date_to:
        hi = str(date_to).strip()
    return lo, hi


def _match_date_query(lo, hi) -> dict:
    """Build a MatchDate range fragment for a Mongo query (ISO strings sort lexicographically)."""
    cond = {}
    if lo:
        cond["$gte"] = lo
    if hi:
        cond["$lte"] = hi
    return {"MatchDate": cond} if cond else {}


def _goals(match):
    """Full-time (home, away) goals as ints, or (None, None) if either is missing."""
    fh, fa = match.get("FTHome"), match.get("FTAway")
    if fh is None or fa is None:
        return None, None
    try:
        return int(fh), int(fa)
    except (ValueError, TypeError):
        return None, None


_RESULT_KEY = {"win": "wins", "draw": "draws", "loss": "losses"}


def _outcome(gf, ga):
    if gf > ga:
        return "win"
    if gf < ga:
        return "loss"
    return "draw"


def _blank_record():
    return {"played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0}


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
    Returns the sorted list of team names that can actually be predicted — i.e. teams that
    have a usable Elo rating — intended for populating a dropdown in the frontend.

    Teams without any Elo are deliberately excluded. Prediction needs a non-NaN Elo, and a
    team like FCSB (absent from EloRatings, blank Elo in Matches.csv) would otherwise produce
    a degraded, Elo-less prediction. A team is considered predictable when it has at least one
    match row carrying a non-blank Elo — exactly the condition under which the predictor
    resolves a real rating. Reads MongoDB first (populated by POST /data/import), falling back
    to the Kaggle dataset if MongoDB is empty, so the dropdown stays usable even before import.
    """
    teams = set()

    # 1) Prefer MongoDB — teams with at least one non-null Elo (home or away appearance).
    #    Imported rows store blank Elo as null (see _clean_record), so $ne: None filters them.
    try:
        db = get_db()
        teams.update(db.matches.distinct("HomeTeam", {"HomeElo": {"$ne": None}}))
        teams.update(db.matches.distinct("AwayTeam", {"AwayElo": {"$ne": None}}))
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
                os.path.join(path, "Matches.csv"),
                usecols=["HomeTeam", "AwayTeam", "HomeElo", "AwayElo"],
            )
            teams.update(matches_df.loc[matches_df["HomeElo"].notna(), "HomeTeam"].tolist())
            teams.update(matches_df.loc[matches_df["AwayElo"].notna(), "AwayTeam"].tolist())
        except Exception as e:
            logger.error("Failed to load teams from Kaggle: %s", str(e))
            print(f"Failed to load teams from Kaggle: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Failed to load team list: {e}")

    # Drop NaN/empty, normalize to str, sort
    cleaned = sorted({
        str(t).strip() for t in teams
        if t is not None and not (isinstance(t, float) and math.isnan(t)) and str(t).strip()
    })
    logger.info("Returning %d teams (with Elo)", len(cleaned))
    print(f"Returning {len(cleaned)} teams (with Elo)")
    return TeamsResponse(teams=cleaned)


@router.post("/matches")
async def get_matches(params: MatchQueryParams):
    """
    Query matches stored in MongoDB with optional filters.

    Filters: home_team / away_team (case-insensitive), season (e.g. '2024/25'),
    and/or an explicit date window (date_from / date_to, ISO YYYY-MM-DD). There is no
    stored Season field, so season is translated to a MatchDate range. Results are sorted
    by match date ('order': 'desc' default, or 'asc') BEFORE the limit is applied.
    """
    db = get_db()
    query = {}

    if params.home_team or params.away_team:
        name_map = _team_name_map(db)
        if params.home_team:
            query["HomeTeam"] = _resolve_team(db, params.home_team, name_map) or params.home_team
        if params.away_team:
            query["AwayTeam"] = _resolve_team(db, params.away_team, name_map) or params.away_team

    try:
        lo, hi = _date_range(params.season, params.date_from, params.date_to)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    query.update(_match_date_query(lo, hi))

    order = (params.order or "desc").lower()
    if order not in ("asc", "desc"):
        return JSONResponse(status_code=400, content={"error": "order must be 'asc' or 'desc'"})
    direction = 1 if order == "asc" else -1

    logger.info("Querying matches: %s (order=%s, limit=%d)", query, order, params.limit)
    print(f"Querying matches: {query} (order={order}, limit={params.limit})")
    cursor = db.matches.find(query, {"_id": 0}).sort("MatchDate", direction)
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


@router.get("/seasons")
async def get_seasons():
    """
    Returns the sorted list of seasons present in the data, e.g. ['2000/01', ..., '2024/25'].

    Seasons are derived from match dates (July cutoff, matching training's get_season), since
    there is no stored Season field. The chatbot can use this to learn the valid season range.
    """
    db = get_db()
    dates = db.matches.distinct("MatchDate")
    seasons = sorted({s for s in (_season_of(d) for d in dates if d) if s})
    logger.info("Returning %d seasons", len(seasons))
    print(f"Returning {len(seasons)} seasons")
    return {"seasons": seasons}


@router.post("/team-summary")
async def team_summary(params: TeamSummaryParams):
    """
    A single team's record (computed server-side) over a season, a date window, or the whole
    dataset. Answers wins/draws/losses, goals for/against, points, clean sheets, and the
    home/away split — without dumping raw rows.
    """
    if not params.team or not str(params.team).strip():
        return JSONResponse(status_code=400, content={"error": "Field 'team' is required."})

    db = get_db()
    canonical = _resolve_team(db, params.team) or str(params.team).strip()

    try:
        lo, hi = _date_range(params.season, params.date_from, params.date_to)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    query = {"$or": [{"HomeTeam": canonical}, {"AwayTeam": canonical}]}
    query.update(_match_date_query(lo, hi))

    overall = _blank_record()
    home = _blank_record()
    away = _blank_record()
    clean_sheets = 0

    projection = {"_id": 0, "HomeTeam": 1, "AwayTeam": 1, "FTHome": 1, "FTAway": 1}
    for m in db.matches.find(query, projection):
        fh, fa = _goals(m)
        if fh is None:
            continue
        if m.get("HomeTeam") == canonical:
            gf, ga, bucket = fh, fa, home
        else:
            gf, ga, bucket = fa, fh, away
        key = _RESULT_KEY[_outcome(gf, ga)]
        for rec in (overall, bucket):
            rec["played"] += 1
            rec["goals_for"] += gf
            rec["goals_against"] += ga
            rec[key] += 1
        if ga == 0:
            clean_sheets += 1

    points = overall["wins"] * 3 + overall["draws"]
    logger.info("team-summary %s: %d played", canonical, overall["played"])
    print(f"team-summary {canonical}: {overall['played']} played")
    return {
        "team": canonical,
        "season": params.season,
        "played": overall["played"],
        "wins": overall["wins"],
        "draws": overall["draws"],
        "losses": overall["losses"],
        "goals_for": overall["goals_for"],
        "goals_against": overall["goals_against"],
        "goal_difference": overall["goals_for"] - overall["goals_against"],
        "points": points,
        "clean_sheets": clean_sheets,
        "home": home,
        "away": away,
    }


@router.post("/head-to-head")
async def head_to_head(params: HeadToHeadParams):
    """
    The aggregated record between two teams (computed server-side): total meetings, each
    team's wins, draws, goals, plus up to `limit` most-recent meetings.
    """
    if not params.team_a or not str(params.team_a).strip() or not params.team_b or not str(params.team_b).strip():
        return JSONResponse(status_code=400, content={"error": "Fields 'team_a' and 'team_b' are required."})

    db = get_db()
    name_map = _team_name_map(db)
    team_a = _resolve_team(db, params.team_a, name_map) or str(params.team_a).strip()
    team_b = _resolve_team(db, params.team_b, name_map) or str(params.team_b).strip()

    try:
        lo, hi = _date_range(params.season, params.date_from, params.date_to)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    query = {
        "$or": [
            {"HomeTeam": team_a, "AwayTeam": team_b},
            {"HomeTeam": team_b, "AwayTeam": team_a},
        ]
    }
    query.update(_match_date_query(lo, hi))

    meetings = team_a_wins = team_b_wins = draws = 0
    team_a_goals = team_b_goals = 0
    recent = []

    projection = {"_id": 0, "MatchDate": 1, "HomeTeam": 1, "AwayTeam": 1, "FTHome": 1, "FTAway": 1}
    for m in db.matches.find(query, projection).sort("MatchDate", -1):
        fh, fa = _goals(m)
        if fh is None:
            continue
        meetings += 1
        if m.get("HomeTeam") == team_a:
            a_goals, b_goals = fh, fa
        else:
            a_goals, b_goals = fa, fh
        team_a_goals += a_goals
        team_b_goals += b_goals
        if a_goals > b_goals:
            team_a_wins += 1
        elif b_goals > a_goals:
            team_b_wins += 1
        else:
            draws += 1
        if params.limit and len(recent) < params.limit:
            recent.append({
                "date": m.get("MatchDate"),
                "home_team": m.get("HomeTeam"),
                "away_team": m.get("AwayTeam"),
                "home_goals": fh,
                "away_goals": fa,
            })

    logger.info("head-to-head %s vs %s: %d meetings", team_a, team_b, meetings)
    print(f"head-to-head {team_a} vs {team_b}: {meetings} meetings")
    return {
        "team_a": team_a,
        "team_b": team_b,
        "meetings": meetings,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "draws": draws,
        "team_a_goals": team_a_goals,
        "team_b_goals": team_b_goals,
        "recent": recent,
    }


@router.post("/standings")
async def standings(params: StandingsParams):
    """
    League table for a season, computed server-side (3 pts win / 1 draw). The dataset spans
    many leagues, so a season can contain several Division codes; in that case 'league' is
    required. Returns the ordered table (points, then goal difference, then goals for).
    """
    if not params.season or not str(params.season).strip():
        return JSONResponse(status_code=400, content={"error": "Field 'season' is required."})

    db = get_db()
    try:
        lo, hi = _season_bounds(params.season)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    base = _match_date_query(lo, hi)

    # Determine which league(s) the season covers; require 'league' if more than one.
    league = params.league.strip() if params.league and params.league.strip() else None
    if league is None:
        divisions = [d for d in db.matches.distinct("Division", base) if d]
        if len(divisions) > 1:
            return JSONResponse(status_code=400, content={
                "error": "This season spans multiple leagues; specify 'league'.",
                "available_leagues": sorted(str(d) for d in divisions),
            })
        league = divisions[0] if divisions else None

    query = dict(base)
    if league is not None:
        query["Division"] = league

    table = {}

    def _row(team):
        if team not in table:
            table[team] = {"team": team, **_blank_record()}
        return table[team]

    projection = {"_id": 0, "HomeTeam": 1, "AwayTeam": 1, "FTHome": 1, "FTAway": 1}
    for m in db.matches.find(query, projection):
        fh, fa = _goals(m)
        if fh is None:
            continue
        ht, at = m.get("HomeTeam"), m.get("AwayTeam")
        if not ht or not at:
            continue
        home_row, away_row = _row(ht), _row(at)
        home_row["played"] += 1
        away_row["played"] += 1
        home_row["goals_for"] += fh
        home_row["goals_against"] += fa
        away_row["goals_for"] += fa
        away_row["goals_against"] += fh
        if fh > fa:
            home_row["wins"] += 1
            away_row["losses"] += 1
        elif fa > fh:
            away_row["wins"] += 1
            home_row["losses"] += 1
        else:
            home_row["draws"] += 1
            away_row["draws"] += 1

    rows = []
    for r in table.values():
        r["goal_difference"] = r["goals_for"] - r["goals_against"]
        r["points"] = r["wins"] * 3 + r["draws"]
        rows.append(r)
    rows.sort(key=lambda r: (-r["points"], -r["goal_difference"], -r["goals_for"], r["team"]))
    for i, r in enumerate(rows, start=1):
        r["position"] = i

    ordered = [
        {
            "position": r["position"],
            "team": r["team"],
            "played": r["played"],
            "wins": r["wins"],
            "draws": r["draws"],
            "losses": r["losses"],
            "goals_for": r["goals_for"],
            "goals_against": r["goals_against"],
            "goal_difference": r["goal_difference"],
            "points": r["points"],
        }
        for r in rows
    ]
    logger.info("standings season=%s league=%s: %d teams", params.season, league, len(ordered))
    print(f"standings season={params.season} league={league}: {len(ordered)} teams")
    return {"season": params.season, "league": league, "table": ordered}
