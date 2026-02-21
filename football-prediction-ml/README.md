# ⚽ MODELUL DE PREDICȚIE A MECIURILOR DE FOTBAL

Un proiect de machine learning care ghicește cine va câștiga meciurile de fotbal folosind date istorice (2000-2025).

## 📥 INSTALARE

### 1. Cerințe Preliminare

- Python 3.7 sau mai nou
- pip (managerul de pachete Python)

### 2. Pași de Instalare

**PASUL 1: Instalează dependințele**

```bash
pip install kagglehub
pip install scikit-learn
pip install matplotlib
pip install seaborn
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

## 📁 Structura Proiectului

```
football-prediction-ml/
├── main.py                          # Scriptul principal de antrenament
├── predict_match.py                 # Utilitate pentru predicții
├── requirements.txt                 # Lista dependințelor
├── README.md                        # Acest fișier (documentație)
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

## 📊 Performanța Modelelor

Proiectul antreneaza și compara două modele:

1. **Random Forest Classifier** (Pădurea Aleatorie)

   - 🌲 Mai bun pentru genul acesta de predicții
   - Capturează relații complexe și nelineare
   - Robust la variații de date
2. **Logistic Regression** (Regresie Logistică)

   - 📖 Model mai simplu și ușor de înțeles
   - Bun pentru bază de comparație
   - Mai rapid la antrenament

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
