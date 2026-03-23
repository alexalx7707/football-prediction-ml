# ⚽ Football Prediction ML — API Documentation

Base URL: `http://localhost:8000`

Interactive docs available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Table of Contents

- [General](#general)
  - [GET /](#get-)
  - [GET /health](#get-health)
- [Predictions](#predictions)
  - [POST /predict/](#post-predict)
- [Training](#training)
  - [POST /train/](#post-train)
  - [GET /train/status](#get-trainstatus)
- [Data (MongoDB)](#data-mongodb)
  - [POST /data/import](#post-dataimport)
  - [GET /data/stats](#get-datastats)
  - [POST /data/matches](#post-datamatches)
  - [POST /data/elo](#post-dataelo)

---

## General

### GET /

Returns API information and a list of available endpoints.

**Response:**
```json
{
  "name": "⚽ Football Prediction ML API",
  "version": "1.0.0",
  "description": "API pentru antrenarea și predicția meciurilor de fotbal",
  "endpoints": {
    "health": "/health (GET)",
    "predict": "/predict (POST)",
    "train": "/train (POST)",
    "training_status": "/train/status (GET)",
    "data_import": "/data/import (POST)",
    "data_stats": "/data/stats (GET)",
    "data_matches": "/data/matches (GET)",
    "data_elo": "/data/elo (GET)",
    "docs": "/docs (Swagger UI)",
    "redoc": "/redoc (ReDoc)"
  }
}
```

---

### GET /health

Checks if the server is running and whether ML models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "api_version": "1.0.0"
}
```

| Field          | Type    | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `status`       | string  | `"healthy"` if models loaded, `"degraded"` otherwise |
| `models_loaded`| boolean | Whether the ML models are available                  |
| `api_version`  | string  | Current API version                                  |

---

## Predictions

### POST /predict/

Makes a prediction for a football match using the trained Random Forest model.

> **Prerequisite:** Models must be trained first via `POST /train/`.

**Request Body (JSON):**

| Field       | Type   | Required | Description                          |
|-------------|--------|----------|--------------------------------------|
| `home_team` | string | Yes      | Home team name (e.g. "Manchester United") |
| `away_team` | string | Yes      | Away team name (e.g. "Liverpool")    |

**Example Request:**
```json
{
  "home_team": "Manchester United",
  "away_team": "Liverpool"
}
```

**Response:**
```json
{
  "match": "Manchester United vs Liverpool",
  "prediction": "🏠 Câștigă ACASĂ",
  "confidence": 76.5,
  "home_team": "Manchester United",
  "away_team": "Liverpool",
  "home_elo": 1823.5,
  "away_elo": 1756.2,
  "elo_difference": 67.3,
  "probabilities": {
    "1": 76.5,
    "0": 16.2,
    "-1": 7.3
  },
  "home_win_prob": 76.5,
  "draw_prob": 16.2,
  "away_win_prob": 7.3
}
```

| Field            | Type   | Description                                      |
|------------------|--------|--------------------------------------------------|
| `match`          | string | Match description                                |
| `prediction`     | string | Predicted outcome with emoji                     |
| `confidence`     | float  | Confidence percentage (0-100)                    |
| `home_team`      | string | Home team name                                   |
| `away_team`      | string | Away team name                                   |
| `home_elo`       | float  | Home team Elo rating                             |
| `away_elo`       | float  | Away team Elo rating                             |
| `elo_difference` | float  | Elo difference (home - away)                     |
| `probabilities`  | object | Probability for each outcome (`"1"`, `"0"`, `"-1"`) |
| `home_win_prob`  | float  | Home win probability (%)                         |
| `draw_prob`      | float  | Draw probability (%)                             |
| `away_win_prob`  | float  | Away win probability (%)                         |

**Prediction values:**
- `1` = Home win
- `0` = Draw
- `-1` = Away win

**Error Responses:**

| Code | Condition                          |
|------|------------------------------------|
| 400  | Invalid team name or processing error |
| 503  | Models not trained yet              |

---

## Training

### POST /train/

Starts model training in the background. Downloads data from Kaggle, preprocesses it, trains Random Forest, Logistic Regression, and XGBoost models, then saves them.

**Request Body (JSON):**

| Field          | Type   | Required | Default    | Description                          |
|----------------|--------|----------|------------|--------------------------------------|
| `test_season`  | string | No       | `"2024/25"`| Season to use for testing/backtest   |
| `n_estimators` | int    | No       | `300`      | Number of XGBoost estimators         |

**Example Request:**
```json
{
  "test_season": "2024/25",
  "n_estimators": 300
}
```

**Response:**
```json
{
  "message": "Antrenamentul a început în background...",
  "status": "started",
  "accuracy_rf": 0.0,
  "accuracy_lr": 0.0,
  "accuracy_xgb": 0.0,
  "test_matches": 0,
  "models_saved": false
}
```

> Training runs in the background. Use `GET /train/status` to monitor progress.

**Error Responses:**

| Code | Condition                       |
|------|---------------------------------|
| 409  | Training already in progress    |

---

### GET /train/status

Returns the current status of the training process.

**Response:**
```json
{
  "status": "completed",
  "message": "Antrenament completat cu succes!"
}
```

**Possible status values:**

| Status          | Description                          |
|-----------------|--------------------------------------|
| `idle`          | No training started                  |
| `starting`      | Training is initializing             |
| `downloading`   | Downloading dataset from Kaggle      |
| `preprocessing` | Preprocessing and cleaning data      |
| `features`      | Feature engineering                   |
| `rolling`       | Calculating rolling features          |
| `splitting`     | Splitting train/test data             |
| `training`      | Training the ML models               |
| `evaluation`    | Evaluating model accuracy             |
| `saving`        | Saving models to disk                 |
| `completed`     | Training finished successfully        |
| `error`         | An error occurred (see `message`)     |

---

## Data (MongoDB)

These endpoints manage importing the Kaggle dataset into MongoDB and querying the stored data.

> **Prerequisite:** MongoDB must be running on `localhost:27017`.
>
> **Database:** `footbal_prediction`

---

### POST /data/import

Downloads the Kaggle dataset and imports all matches and Elo ratings into MongoDB. Existing collections are **dropped and recreated** each time.

**Request Body:** None

**Response:**
```json
{
  "message": "Dataset imported successfully into MongoDB",
  "matches_imported": 185432,
  "elo_ratings_imported": 52891
}
```

| Field                 | Type   | Description                        |
|-----------------------|--------|------------------------------------|
| `message`             | string | Status message                     |
| `matches_imported`    | int    | Number of match records imported   |
| `elo_ratings_imported`| int    | Number of Elo rating records imported |

**Indexes created automatically:**

| Collection    | Indexed Fields                        |
|---------------|---------------------------------------|
| `matches`     | `HomeTeam`, `AwayTeam`, `MatchDate`, `Season` |
| `elo_ratings` | `club`, `date`                        |

**Error Responses:**

| Code | Condition                      |
|------|--------------------------------|
| 404  | Matches.csv not found in dataset |
| 502  | Failed to download from Kaggle  |

---

### GET /data/stats

Returns document counts for each collection in the MongoDB database.

**Response:**
```json
{
  "collections": [
    {
      "collection": "elo_ratings",
      "document_count": 52891
    },
    {
      "collection": "matches",
      "document_count": 185432
    }
  ]
}
```

---

### POST /data/matches

Query match records stored in MongoDB with optional filters.

**Request Body (JSON):**

| Field       | Type   | Required | Default | Description                          |
|-------------|--------|----------|---------|--------------------------------------|
| `home_team` | string | No       | `null`  | Filter by home team name             |
| `away_team` | string | No       | `null`  | Filter by away team name             |
| `season`    | string | No       | `null`  | Filter by season (e.g. `"2024/25"`)  |
| `limit`     | int    | No       | `50`    | Max results (1-1000)                 |

**Example Requests:**
```json
{}
```
```json
{
  "home_team": "Manchester United",
  "limit": 10
}
```
```json
{
  "season": "2024/25",
  "limit": 20
}
```
```json
{
  "home_team": "Liverpool",
  "away_team": "Manchester United",
  "season": "2024/25"
}
```

**Response:** Array of match objects
```json
[
  {
    "MatchDate": "2024-10-20",
    "Season": "2024/25",
    "HomeTeam": "Manchester United",
    "AwayTeam": "Liverpool",
    "FTHome": 1,
    "FTAway": 2,
    "FTResult": "A",
    "HomeElo": 1823.5,
    "AwayElo": 1856.2,
    "HomeShots": 12,
    "AwayShots": 15,
    "..."
  }
]
```

---

### POST /data/elo

Query Elo ratings stored in MongoDB, sorted by date descending (most recent first).

**Request Body (JSON):**

| Field   | Type   | Required | Default | Description                    |
|---------|--------|----------|---------|--------------------------------|
| `club`  | string | No       | `null`  | Filter by club name            |
| `limit` | int    | No       | `50`    | Max results (1-1000)           |

**Example Requests:**
```json
{}
```
```json
{
  "club": "Barcelona",
  "limit": 10
}
```

**Response:** Array of Elo rating objects
```json
[
  {
    "club": "Barcelona",
    "date": "2025-03-15",
    "elo": 1892.3
  }
]
```

---

## Typical Workflow

```
1. POST /train/                → Start training (runs in background)
2. GET  /train/status          → Poll until status = "completed"
3. POST /predict/              → Make match predictions
4. POST /data/import           → Import dataset into MongoDB
5. POST /data/matches          → Query stored matches
6. POST /data/elo              → Query Elo ratings
```

---

## Error Format

All errors follow this structure:
```json
{
  "detail": "Error message describing what went wrong"
}
```

| HTTP Code | Meaning                              |
|-----------|--------------------------------------|
| 400       | Bad request / invalid input          |
| 404       | Resource not found                   |
| 409       | Conflict (e.g. training in progress) |
| 502       | Upstream error (e.g. Kaggle download)|
| 503       | Service unavailable (models not loaded)|
