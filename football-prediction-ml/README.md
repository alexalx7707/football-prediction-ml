# ⚽ MODELUL DE PREDICȚIE A MECIURILOR DE FOTBAL

Un proiect de machine learning care ghicește cine va câștiga meciurile de fotbal folosind date istorice (2000-2025).

## 📥 INSTALARE

### 1. Cerințe Preliminare

- Python 3.7 sau mai nou
- pip (managerul de pachete Python)

### 2. Pași de Instalare

**PASUL 1: Instalează dependințele**

```bash
pip install -r requirements.txt
```

Sau manual:

```bash
pip install kagglehub
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install xgboost
pip install fastapi uvicorn
```

**PASUL 2: Configurare Kaggle API** (Necesar pentru a descărca datele)

a) Mergi la https://www.kaggle.com/settings/account
b) Click "Create New API Token" - asta descarcă fișierul `kaggle.json`
c) Mută fișierul în `C:\Users\<TuNume>\.kaggle\kaggle.json` (Windows)
d) Rulează: `icacls "C:\Users\<TuNume>\.kaggle\kaggle.json" /inheritance:r /grant:r "%username%:F"` pentru permisiuni

### 3. Rularea Proiectului

**Antrenează modelul:**

```bash
python main.py
```

Asta va:

- Descarca datele meciurilor din Kaggle
- Preprocesa și analiza datele
- Antrena două modele ML (Random Forest & Regresie Logistică)
- Evalua performanța modelelor
- Salva modelele antrenate în folder-ul `models/`
- Genera grafice frumoase cu rezultatele

**Fă predicții:**

```bash
python predict_match.py "Manchester United" "Liverpool"
```

## 📦 Pachete Necesare

| Pachet       | Versie | Scop                               |
| ------------ | ------ | ---------------------------------- |
| kagglehub    | 0.2.4  | Descarcă dataset-uri de pe Kaggle |
| pandas       | 2.1.3  | Manipulare și analiză de date    |
| scikit-learn | 1.3.2  | Algoritmi de machine learning      |
| numpy        | 1.24.3 | Calcule numerice                   |
| matplotlib   | 3.8.2  | Vizualizare date                   |
| seaborn      | 0.13.0 | Grafice avansate                   |
| joblib       | 1.3.2  | Salvare și încarcă modele       |
| xgboost      | 2.0.0  | Gradient Boosting - Model ML performant ⚡ |
| fastapi      | 0.104.1| Framework pentru API REST          |
| uvicorn      | 0.24.0 | Server ASGI pentru FastAPI         |
| pydantic     | 2.5.0  | Validare și structura datelor      |
| pymongo      | 4.6.1  | Driver MongoDB pentru Python       |

## 🗄️ MongoDB Setup

Proiectul folosește MongoDB pentru stocarea datelor din dataset (meciuri și ratinguri Elo).

### 1. Instalare MongoDB

**Windows:**
- Descarcă MongoDB Community Server de la https://www.mongodb.com/try/download/community
- Rulează installerul și urmele pașii (include MongoDB Compass opțional)
- MongoDB va porni automat ca serviciu Windows

**Sau cu Docker:**
```bash
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

### 2. Verificare conexiune

MongoDB trebuie să ruleze pe `localhost:27017`. Verifică cu:
```bash
mongosh --eval "db.runCommand({ ping: 1 })"
```

### 3. Baza de date

- **Nume:** `footbal_prediction`
- **Connection string:** `mongodb://localhost:27017/footbal_prediction`
- Se creează automat la primul import de date

### 4. Colecții

| Colecție      | Descriere                              | Indexuri                              |
|---------------|----------------------------------------|---------------------------------------|
| `matches`     | Toate meciurile din dataset (2000-2025)| HomeTeam, AwayTeam, MatchDate, Season |
| `elo_ratings` | Ratinguri Elo pe echipă și dată       | club, date                            |

### 5. Import date în MongoDB

După pornirea serverului API:
```bash
curl -X POST http://localhost:8000/data/import
```

Aceasta va descărca dataset-ul Kaggle și va importa toate datele în MongoDB.

## 📁 Structura Proiectului

```
football-prediction-ml/
├── main.py                          # Scriptul principal de antrenament
├── predict_match.py                 # Utilitate pentru predicții
├── api.py                           # FastAPI server (punct de intrare)
├── db.py                            # Configurare conexiune MongoDB
├── requirements.txt                 # Lista dependințelor
├── README.md                        # Acest fișier (documentație)
├── API.md                           # Documentație detaliată API
├── routers/                         # Routere FastAPI (endpoint-uri separate)
│   ├── __init__.py
│   ├── prediction.py                # Endpoint-uri predicție
│   ├── training.py                  # Endpoint-uri antrenament
│   └── data.py                      # Endpoint-uri MongoDB (import/query)
└── models/                          # Modele salvate (creat după rularea main.py)
    ├── random_forest_model.pkl
    ├── logistic_regression_model.pkl
    ├── feature_columns.pkl
    ├── confusion_matrices.png
    └── feature_importance.png
```

## 🤖 Cum Funcționează

### Caracteristici ale Modelului

- **Elo Acasă**: Rating-ul echipei de acasă
- **Elo Deplasare**: Rating-ul echipei din deplasare
- **Forme Recent**: Cum au jucat în ultimele 3-5 meciuri
- **Statistici**: Șuturi, colțuri, cartonașe, etc.

### Ce Prezice Modelul

- **1** = Câștigă ACASĂ
- **0** = Se termină EGAL
- **-1** = Câștigă DEPLASARE

Fiecare predicție vine cu o procentaj de încredere (0-100%).

## 💻 Exemplu de Utilizare

```python
# Din linia de comandă:
python predict_match.py "Manchester United" "Liverpool"

# Sau din cod:
from predict_match import predict_match_by_teams

result = predict_match_by_teams("Manchester United", "Liverpool")
print(f"Predicție: {result['prediction']}")
print(f"Încredere: {result['confidence']:.2f}%")
```

## 🌐 API REST (OpenAPI/Swagger)

Puteți antrena modelul și face predicții prin HTTP API.

> **📖 Documentație completă API cu toate endpoint-urile, parametrii și exemplele: [API.md](API.md)**

### Pornire API Server

```bash
python api.py
```

Server va porni pe `http://localhost:8000`

### Documentație API (Swagger UI)
- **Interactive Docs:** http://localhost:8000/docs
- **ReDoc Docs:** http://localhost:8000/redoc

### Endpoints Disponibile

| Metodă | Endpoint           | Descriere                          |
|--------|--------------------|------------------------------------|
| GET    | `/`                | Informații despre API               |
| GET    | `/health`          | Health check                        |
| POST   | `/predict/`        | Predicție meci                     |
| POST   | `/train/`          | Antrenează modelele                |
| GET    | `/train/status`    | Status antrenament                  |
| POST   | `/data/import`     | Importă dataset în MongoDB         |
| GET    | `/data/stats`      | Statistici colecții MongoDB        |
| POST   | `/data/matches`    | Interoghează meciuri din MongoDB   |
| POST   | `/data/elo`        | Interoghează ratinguri Elo          |

### Exemplu rapid

```bash
# 1. Antrenează modelul
curl -X POST http://localhost:8000/train/ -H "Content-Type: application/json" -d "{\"test_season\": \"2024/25\"}"

# 2. Verifică status antrenament
curl http://localhost:8000/train/status

# 3. Predicție meci
curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d "{\"home_team\": \"Manchester United\", \"away_team\": \"Liverpool\"}"

# 4. Importă date în MongoDB
curl -X POST http://localhost:8000/data/import
```

### Postman - Exemple gata de folosit

Poți testa API-ul direct din Postman cu request-urile de mai jos.

1. Creează un Environment nou în Postman, de exemplu `Football Local`.
2. Adaugă variabila `baseUrl` cu valoarea `http://localhost:8000`.
3. Pentru request-urile `POST` cu body JSON, setează header-ul `Content-Type: application/json`.

#### 1) Health Check

- **Method:** `GET`
- **URL:** `{{baseUrl}}/health`
- **Body:** nu este necesar

#### 2) Informații API

- **Method:** `GET`
- **URL:** `{{baseUrl}}/`
- **Body:** nu este necesar

#### 3) Pornește antrenamentul

- **Method:** `POST`
- **URL:** `{{baseUrl}}/train/`
- **Body (raw / JSON):**

```json
{
   "test_season": "2024/25",
   "n_estimators": 300
}
```

#### 4) Verifică status antrenament

- **Method:** `GET`
- **URL:** `{{baseUrl}}/train/status`
- **Body:** nu este necesar

Răspuns posibil (după finalizare):

```json
{
   "status": "completed",
   "message": "Antrenament completat cu succes!"
}
```

#### 5) Predicție meci

- **Method:** `POST`
- **URL:** `{{baseUrl}}/predict/`
- **Body (raw / JSON):**

```json
{
   "home_team": "Manchester United",
   "away_team": "Liverpool"
}
```

Notă: endpoint-ul poate returna `503` dacă modelele nu sunt încă antrenate.

#### 6) Import dataset în MongoDB

- **Method:** `POST`
- **URL:** `{{baseUrl}}/data/import`
- **Body:** nu este necesar

#### 7) Statistici colecții MongoDB

- **Method:** `GET`
- **URL:** `{{baseUrl}}/data/stats`
- **Body:** nu este necesar

#### 8) Query meciuri din MongoDB

- **Method:** `POST`
- **URL:** `{{baseUrl}}/data/matches`
- **Body (raw / JSON):**

```json
{
   "home_team": "Manchester United",
   "away_team": "Liverpool",
   "season": "2024/25",
   "limit": 10
}
```

Toate câmpurile sunt opționale; poți trimite și body gol:

```json
{
   "limit": 5
}
```

#### 9) Query ratinguri Elo

- **Method:** `POST`
- **URL:** `{{baseUrl}}/data/elo`
- **Body (raw / JSON):**

```json
{
   "club": "Manchester United",
   "limit": 10
}
```

Și aici câmpurile sunt opționale; pentru ultimele ratinguri indiferent de club:

```json
{
   "limit": 20
}
```

#### Ordine recomandată în Postman

1. `GET {{baseUrl}}/health`
2. `POST {{baseUrl}}/train/`
3. `GET {{baseUrl}}/train/status` (până vezi `completed`)
4. `POST {{baseUrl}}/predict/`
5. `POST {{baseUrl}}/data/import`
6. `GET {{baseUrl}}/data/stats`
7. `POST {{baseUrl}}/data/matches`
8. `POST {{baseUrl}}/data/elo`

## 📊 Performanța Modelelor

Proiectul antreneaza și compara trei modele:

1. **Random Forest Classifier** (Pădurea Aleatorie)

   - 🌲 Mai bun pentru genul acesta de predicții
   - Capturează relații complexe și nelineare
   - Robust la variații de date

2. **Logistic Regression** (Regresie Logistică)

   - 📖 Model mai simplu și ușor de înțeles
   - Bun pentru bază de comparație
   - Mai rapid la antrenament

3. **XGBoost** (Extreme Gradient Boosting) ⚡

   - 🚀 **De 2-3x mai performant** decât Random Forest
   - Algoritm moderna de Gradient Boosting
   - Optimizat pentru predicții cu date complexe
   - Ideal pentru competiții ML și predicții de fotbal
   - Consumă mai puțin memorie cu performanță mai bună

### Metrici de Evaluare

- **Acuratețe** - De câte ori ghicește corect
- **Precizie** - Când spune ceva, cât de adevărat e
- **Sensibilitate** - Nu ratează rezultate importante
- **F1-Score** - Balanța perfectă între precizie și sensibilitate

## 🔧 Depanare

### Eroare Kaggle API

Dacă primești eroare cu Kaggle:

1. Verifică că `kaggle.json` e în `~/.kaggle/`
2. Setează permisiuni corecte
3. Testează: `kaggle datasets download -d adamgbor/club-football-match-data-2000-2025`

### Echipa Nu Este Găsită

Dacă primești "Team not found":

- Numele echipei trebuie să se potrivească exact cu cel din dataset
- Unele variații exista (ex: "Manchester United" vs "Man United")
- Verifică output-ul din `main.py` pentru echipe disponibile

### Probleme de Memorie

Dacă programul se blochează:

- Scade `n_estimators` în Random Forest
- Folosește mai puține date
- Rulează pe calculator cu mai multă RAM

## 🚀 Îmbunătățiri Viitoare

Posibile extensii ale proiectului:

- 👥 Statistici jucători individuali
- 📈 Forma echipei (ultimele meciuri)
- 🏠 Avantajul jucării acasă

Pentru probleme cu modelul:

- Dataset Kaggle: https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025
- Documentație scikit-learn: https://scikit-learn.org/
- Python docs: https://docs.python.org/
