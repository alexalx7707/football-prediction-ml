"""
🏟️ UTILITATE DE PREDICȚIE A MECIURILOR DE FOTBAL
Ruleaza asta ca să ghicești cine va câștiga! (Nu garantez, fotbalul e imprevizibil!)
"""

import joblib
import numpy as np
import pandas as pd
import sys
import os

# 📥 Încarcă dataset-ul original pentru ratingurile Elo ale echipelor
def load_dataset():
    """Încarcă dataset-ul original de meciuri din Kaggle"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
        df = pd.read_csv(os.path.join(path, 'Matches.csv'))
        return df
    except:
        return None  # Dacă nu reușim, nu-i problemă

def get_team_elo(team_name, df, is_home=True):
    """Ia ratingul Elo cel mai recent pentru o echipă (de pe piața asta!)"""
    if df is None:
        return 1500  # Elo implicit dacă nu avem date
    
    if is_home:
        matches = df[df['HomeTeam'] == team_name]
        if len(matches) > 0:
            return matches['HomeElo'].iloc[-1]  # Ultimul meci, cel mai recent
    else:
        matches = df[df['AwayTeam'] == team_name]
        if len(matches) > 0:
            return matches['AwayElo'].iloc[-1]  # Ultimul meci, cel mai recent
    
    return 1500  # Dacă nu gasim echipa, dam 1500 (media globala)

def load_models():
    """Încarcă modelele antrenate și lista de caracteristici"""
    try:
        model = joblib.load('models/random_forest_model.pkl')  # Pădurea inteligentă
        feature_columns = joblib.load('models/feature_columns.pkl')  # Ce ingrediente folosim
        return model, feature_columns
    except FileNotFoundError:
        print("❌ EROARE: Nu am găsit modelele! Trebuie să rulezi main.py prima dată!")
        sys.exit(1)

def predict_match_by_teams(home_team, away_team, df=None):
    """
    Ghicește cine va câștiga un meci.
    
    Parametri:
        home_team (string): Echipa de acasă
        away_team (string): Echipa din deplasare
        df (DataFrame): Dataset cu meciuri (pentru a lua ratingurile Elo)
    
    Returnează:
        dict: Predicția cu rezultat și încredere
    """
    model, feature_columns = load_models()
    
    try:
        # 🎯 Luam ratingurile Elo pentru echipe
        home_elo = get_team_elo(home_team, df, is_home=True)
        away_elo = get_team_elo(away_team, df, is_home=False)
        
        # 🔨 Construim vectorul de caracteristici cu ce avem disponibil
        features = []
        for col in feature_columns:
            if col == 'HomeElo':
                features.append(home_elo)  # Ratingul echipei de acasă
            elif col == 'AwayElo':
                features.append(away_elo)  # Ratingul echipei din deplasare
            elif col == 'EloDifference':
                features.append(home_elo - away_elo)  # Cine e mai tare?
            elif col == 'EloTotal':
                features.append(home_elo + away_elo)  # Cat de bun e meciul?
            elif col.startswith('Form') or col.endswith('Diff'):
                features.append(0)  # Form nu avem (presupunem 0)
            elif col.startswith('Shots') or col.startswith('Corners') or col.startswith('Card'):
                features.append(0)  # Statistici nu avem (presupunem 0)
            elif col == 'Year':
                features.append(2026)  # Anul curent (2026 baby!)
            else:
                features.append(0)  # Alte coloane - pun 0
        
        features = np.array([features])  # Transform in array
        
        # 🤖 Facem predicția cu modelul nostru inteligent
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]  # Câtă încredere are?
        
        # 📊 Traduc numarul in rezultat uman
        result_map = {
            1.0: "🏠 Câștigă ACASĂ",
            0.0: "🤝 EGAL",
            -1.0: "🚗 Câștigă DEPLASARE"
        }
        
        # 📈 Luam probabilitățile pentru fiecare caz
        classes = list(model.classes_)
        prob_dict = {classes[i]: probabilities[i] * 100 for i in range(len(classes))}
        
        return {
            'match': f"{home_team} vs {away_team}",  # Meciul
            'prediction': result_map.get(prediction, "❓ ???"),  # Ce cred eu că se întâmplă
            'confidence': max(probabilities) * 100,  # Cât de sigur sunt
            'home_elo': home_elo,
            'away_elo': away_elo,
            'probabilities': prob_dict,
            'home_win_prob': prob_dict.get(1.0, 0),  # Șansă ACASĂ
            'draw_prob': prob_dict.get(0.0, 0),  # Șansă EGAL
            'away_win_prob': prob_dict.get(-1.0, 0),  # Șansă DEPLASARE
        }
    except Exception as e:
        return {
            'error': str(e),
            'home_team': home_team,
            'away_team': away_team
        }

if __name__ == "__main__":
    print("\n🏟️ UTILITATE DE PREDICȚIE FOTBAL")
    print("="*60)
    
    if len(sys.argv) < 3:
        print("\n📖 UTILIZARE: python predict_match.py <echipa_acasa> <echipa_deplasare>")
        print("📌 EXEMPLU: python predict_match.py \"Manchester United\" \"Liverpool\"")
        print("\n⚠️ Atenție: Numele echipelor trebuie să se potrivească cu cele din dataset!")
        print("   (de ex: 'Man United', 'Arsenal', 'Liverpool', etc.)")
        print("\n🔄 Rulam o predicție exemplu...")
        print("-"*60)
        
        # 🔮 Predicție exemplu (două echipe imaginare cu ratinguri diferite)
        df = load_dataset()
        result = predict_match_by_teams("Manchester United", "Liverpool", df)
        
    else:
        home_team = sys.argv[1]  # Prima echipă
        away_team = sys.argv[2]  # A doua echipă
        
        print(f"\n⏳ Încarc dataset-ul...")
        df = load_dataset()
        result = predict_match_by_teams(home_team, away_team, df)
    
    if 'error' in result:
        print(f"\n❌ EROARE: {result['error']}")
        print(f"   Verifică că echipa '{result.get('home_team')}' există în dataset!")
    else:
        print(f"\n🎯 MECIUL: {result['match']}")
        print(f"   Elo ACASĂ: {result['home_elo']:.2f}")
        print(f"   Elo DEPLASARE: {result['away_elo']:.2f}")
        print(f"   Diferență Elo: {result['home_elo'] - result['away_elo']:.2f} (cine e mai tare)")
        print("-"*60)
        print(f"\n🔮 PREDICȚIA: {result['prediction']}")
        print(f"   Încredere: {result['confidence']:.2f}%  (cât de sigur sunt)")
        print(f"\n📊 PROBABILITĂȚI DETALIATE:")
        print(f"   {result['home_win_prob']:.2f}% - Câștigă ACASĂ 🏠")
        print(f"   {result['draw_prob']:.2f}% - Se termină EGAL 🤝")
        print(f"   {result['away_win_prob']:.2f}% - Câștigă DEPLASARE 🚗")
    
    print("="*60 + "\n")
