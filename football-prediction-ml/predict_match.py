"""
🏟️ UTILITATE DE PREDICȚIE A MECIURILOR DE FOTBAL
Ruleaza asta ca să ghicești cine va câștiga! (Nu garantez, fotbalul e imprevizibil!)
"""

import joblib
import numpy as np
import pandas as pd
import sys
import os

# ──────────────────────────────────────────────
# 📥 ÎNCĂRCARE DATE
# ──────────────────────────────────────────────

def load_dataset():
    """Încarcă dataset-ul original de meciuri din Kaggle"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
        df = pd.read_csv(os.path.join(path, 'Matches.csv'))
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        df = df.sort_values('MatchDate')  # important: sortat cronologic
        return df
    except:
        return None

def load_elo_ratings():
    """Încarcă EloRatings.csv pentru ratinguri actualizate"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
        elo = pd.read_csv(os.path.join(path, 'EloRatings.csv'))
        elo['date'] = pd.to_datetime(elo['date'], errors='coerce')
        return elo
    except:
        return None

def load_models():
    """Încarcă modelele antrenate și lista de caracteristici"""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, feature_columns
    except FileNotFoundError:
        print("❌ EROARE: Nu am găsit modelele! Trebuie să rulezi main.py prima dată!")
        sys.exit(1)

# ──────────────────────────────────────────────
# 📊 FUNCȚII PENTRU STATISTICI RECENTE
# ──────────────────────────────────────────────

def get_team_elo(team_name, elo_df=None, matches_df=None, is_home=True):
    """
    Ia cel mai recent rating Elo pentru o echipă.
    Prioritizează EloRatings.csv, fallback pe Matches.csv.
    """
    # 1. Încearcă EloRatings.csv (cel mai precis)
    if elo_df is not None:
        team_elo = elo_df[elo_df['club'] == team_name].sort_values('date')
        if len(team_elo) > 0:
            return float(team_elo['elo'].iloc[-1])

    # 2. Fallback: ultimul meci din Matches.csv
    if matches_df is not None:
        col = 'HomeTeam' if is_home else 'AwayTeam'
        elo_col = 'HomeElo' if is_home else 'AwayElo'
        team_matches = matches_df[matches_df[col] == team_name].sort_values('MatchDate')
        if len(team_matches) > 0:
            return float(team_matches[elo_col].iloc[-1])

    return 1500.0  # fallback global dacă echipa nu e găsită


def get_team_recent_stats(team_name, matches_df, n=5):
    """
    Calculează media statisticilor din ultimele N meciuri ale unei echipe
    (atât acasă cât și în deplasare).

    Returnează un dict cu mediile pentru: Shots, Target, Corners, Fouls, Yellow, Red
    din perspectiva echipei (indiferent dacă a jucat acasă sau deplasare).
    """
    if matches_df is None:
        return {}

    # Meciurile acasă
    home_matches = matches_df[matches_df['HomeTeam'] == team_name].copy()
    home_matches = home_matches.rename(columns={
        'HomeShots': 'Shots', 'HomeTarget': 'Target', 'HomeCorners': 'Corners',
        'HomeFouls': 'Fouls', 'HomeYellow': 'Yellow', 'HomeRed': 'Red',
        'Form3Home': 'Form3', 'Form5Home': 'Form5'
    })

    # Meciurile deplasare
    away_matches = matches_df[matches_df['AwayTeam'] == team_name].copy()
    away_matches = away_matches.rename(columns={
        'AwayShots': 'Shots', 'AwayTarget': 'Target', 'AwayCorners': 'Corners',
        'AwayFouls': 'Fouls', 'AwayYellow': 'Yellow', 'AwayRed': 'Red',
        'Form3Away': 'Form3', 'Form5Away': 'Form5'
    })

    stat_cols = ['MatchDate', 'Shots', 'Target', 'Corners', 'Fouls', 'Yellow', 'Red', 'Form3', 'Form5']
    existing_home = [c for c in stat_cols if c in home_matches.columns]
    existing_away = [c for c in stat_cols if c in away_matches.columns]

    all_matches = pd.concat([
        home_matches[existing_home],
        away_matches[existing_away]
    ]).sort_values('MatchDate')

    if len(all_matches) == 0:
        return {}

    # Ultimele N meciuri
    recent = all_matches.tail(n)

    stats = {}
    for col in ['Shots', 'Target', 'Corners', 'Fouls', 'Yellow', 'Red', 'Form3', 'Form5']:
        if col in recent.columns:
            stats[col] = recent[col].fillna(0).mean()

    return stats


def build_feature_vector(home_team, away_team, matches_df, elo_df, feature_columns):
    """
    Construiește vectorul de caracteristici pentru un meci,
    folosind statistici reale recente în loc de zerouri.
    """
    home_elo = get_team_elo(home_team, elo_df, matches_df, is_home=True)
    away_elo = get_team_elo(away_team, elo_df, matches_df, is_home=False)

    home_stats = get_team_recent_stats(home_team, matches_df)
    away_stats = get_team_recent_stats(away_team, matches_df)

    # Valori derivate
    elo_diff  = home_elo - away_elo
    elo_total = home_elo + away_elo

    features = []
    missing_cols = []

    for col in feature_columns:
        val = None

        # Elo
        if col == 'HomeElo':             val = home_elo
        elif col == 'AwayElo':           val = away_elo
        elif col == 'EloDifference':     val = elo_diff
        elif col == 'EloTotal':          val = elo_total

        # Form
        elif col == 'Form3Home':         val = home_stats.get('Form3', 0)
        elif col == 'Form5Home':         val = home_stats.get('Form5', 0)
        elif col == 'Form3Away':         val = away_stats.get('Form3', 0)
        elif col == 'Form5Away':         val = away_stats.get('Form5', 0)
        elif col == 'Form3Diff':         val = home_stats.get('Form3', 0) - away_stats.get('Form3', 0)
        elif col == 'Form5Diff':         val = home_stats.get('Form5', 0) - away_stats.get('Form5', 0)

        # Shots
        elif col == 'HomeShots':         val = home_stats.get('Shots', 0)
        elif col == 'AwayShots':         val = away_stats.get('Shots', 0)
        elif col == 'ShotsDifference':   val = home_stats.get('Shots', 0) - away_stats.get('Shots', 0)

        # Target
        elif col == 'HomeTarget':        val = home_stats.get('Target', 0)
        elif col == 'AwayTarget':        val = away_stats.get('Target', 0)

        # Corners
        elif col == 'HomeCorners':       val = home_stats.get('Corners', 0)
        elif col == 'AwayCorners':       val = away_stats.get('Corners', 0)
        elif col == 'CornersDifference': val = home_stats.get('Corners', 0) - away_stats.get('Corners', 0)

        # Fouls
        elif col == 'HomeFouls':         val = home_stats.get('Fouls', 0)
        elif col == 'AwayFouls':         val = away_stats.get('Fouls', 0)

        # Cards
        elif col == 'HomeYellow':        val = home_stats.get('Yellow', 0)
        elif col == 'AwayYellow':        val = away_stats.get('Yellow', 0)
        elif col == 'HomeRed':           val = home_stats.get('Red', 0)
        elif col == 'AwayRed':           val = away_stats.get('Red', 0)
        elif col == 'CardPointsHome':    val = home_stats.get('Yellow', 0) + 2 * home_stats.get('Red', 0)
        elif col == 'CardPointsAway':    val = away_stats.get('Yellow', 0) + 2 * away_stats.get('Red', 0)

        # Year
        elif col == 'Year':              val = pd.Timestamp.now().year

        else:
            val = 0
            missing_cols.append(col)

        features.append(val if val is not None else 0)

    if missing_cols:
        print(f"   ⚠️  Coloane necunoscute, puse pe 0: {missing_cols}")

    return np.array([features]), home_elo, away_elo


# ──────────────────────────────────────────────
# 🔮 PREDICȚIE
# ──────────────────────────────────────────────

def predict_match_by_teams(home_team, away_team, matches_df=None, elo_df=None):
    """
    Ghicește cine va câștiga un meci folosind statistici recente reale.

    Parametri:
        home_team  (str): Echipa de acasă
        away_team  (str): Echipa din deplasare
        matches_df: DataFrame cu meciuri istorice
        elo_df:     DataFrame cu ratinguri Elo actualizate

    Returnează:
        dict cu predicția, probabilitățile și statisticile folosite
    """
    model, feature_columns = load_models()

    try:
        features, home_elo, away_elo = build_feature_vector(
            home_team, away_team, matches_df, elo_df, feature_columns
        )

        prediction   = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result_map = {
            1.0:  "🏠 Câștigă ACASĂ",
            0.0:  "🤝 EGAL",
            -1.0: "🚗 Câștigă DEPLASARE"
        }

        classes  = list(model.classes_)
        prob_dict = {classes[i]: probabilities[i] * 100 for i in range(len(classes))}

        # Statistici recente pentru afișare
        home_stats = get_team_recent_stats(home_team, matches_df)
        away_stats = get_team_recent_stats(away_team, matches_df)

        return {
            'match':          f"{home_team} vs {away_team}",
            'prediction':     result_map.get(prediction, "❓ ???"),
            'confidence':     max(probabilities) * 100,
            'home_elo':       home_elo,
            'away_elo':       away_elo,
            'probabilities':  prob_dict,
            'home_win_prob':  prob_dict.get(1.0, 0),
            'draw_prob':      prob_dict.get(0.0, 0),
            'away_win_prob':  prob_dict.get(-1.0, 0),
            'home_stats':     home_stats,
            'away_stats':     away_stats,
        }

    except Exception as e:
        return {
            'error':      str(e),
            'home_team':  home_team,
            'away_team':  away_team
        }


# ──────────────────────────────────────────────
# 🚀 MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏟️ UTILITATE DE PREDICȚIE FOTBAL")
    print("="*60)

    if len(sys.argv) < 3:
        print("\n📖 UTILIZARE: python predict_match.py <echipa_acasa> <echipa_deplasare>")
        print("📌 EXEMPLU:   python predict_match.py \"Manchester United\" \"Liverpool\"")
        print("\n🔄 Rulam o predicție exemplu...")
        print("-"*60)
        home_team = "Manchester United"
        away_team = "Liverpool"
    else:
        home_team = sys.argv[1]
        away_team = sys.argv[2]

    print(f"\n⏳ Încarc dataset-ul și ratingurile Elo...")
    matches_df = load_dataset()
    elo_df     = load_elo_ratings()

    if matches_df is None:
        print("⚠️  Nu am putut încărca dataset-ul - predicțiile vor folosi doar Elo!")
    if elo_df is None:
        print("⚠️  Nu am putut încărca EloRatings.csv - folosesc Elo din Matches.csv")

    result = predict_match_by_teams(home_team, away_team, matches_df, elo_df)

    if 'error' in result:
        print(f"\n❌ EROARE: {result['error']}")
        print(f"   Verifică că echipa '{result.get('home_team')}' există în dataset!")
    else:
        elo_diff = result['home_elo'] - result['away_elo']

        print(f"\n🎯 MECIUL: {result['match']}")
        print(f"   Elo ACASĂ:      {result['home_elo']:.1f}")
        print(f"   Elo DEPLASARE:  {result['away_elo']:.1f}")
        print(f"   Diferență Elo:  {elo_diff:+.1f}")

        # Afișează statisticile recente folosite
        hs = result['home_stats']
        as_ = result['away_stats']
        if hs or as_:
            print(f"\n📈 STATISTICI RECENTE (media ultimelor 5 meciuri):")
            print(f"   {'Stat':<15} {'Acasă':>8} {'Deplasare':>10}")
            print(f"   {'-'*35}")
            for stat, label in [('Shots','Șuturi'), ('Target','Pe poartă'),
                                 ('Corners','Cornere'), ('Fouls','Fault'),
                                 ('Yellow','Galbene'), ('Form5','Form5')]:
                h_val = hs.get(stat, '-')
                a_val = as_.get(stat, '-')
                h_str = f"{h_val:.1f}" if isinstance(h_val, float) else str(h_val)
                a_str = f"{a_val:.1f}" if isinstance(a_val, float) else str(a_val)
                print(f"   {label:<15} {h_str:>8} {a_str:>10}")

        print(f"\n{'─'*60}")
        print(f"🔮 PREDICȚIA:  {result['prediction']}")
        print(f"   Încredere:  {result['confidence']:.1f}%")
        print(f"\n📊 PROBABILITĂȚI:")
        print(f"   {result['home_win_prob']:.1f}%  Câștigă ACASĂ  🏠")
        print(f"   {result['draw_prob']:.1f}%  EGAL            🤝")
        print(f"   {result['away_win_prob']:.1f}%  Câștigă DEPLASARE 🚗")

    print("="*60 + "\n")