# 🏟️ Prognoza meciuri de fotbal - Cod mai plin de viață decât un meci la penalty! 🤦

import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 📥 DESCĂRCARE DATE
# ──────────────────────────────────────────────

print("Descarcă datele meciurilor de fotbal din Kaggle...")
path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
print(f"Datele au sosit în: {path}")

data_files = os.listdir(path)
print(f"\nFișiere disponibile: {data_files}")

matches_path = os.path.join(path, 'Matches.csv')
if os.path.exists(matches_path):
    df = pd.read_csv(matches_path)
    print(f"\n✅ Încarcă fișierul: Matches.csv")
else:
    csv_files = [f for f in data_files if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        print(f"\n✅ Am încărcat: {csv_files[0]}")
    else:
        raise FileNotFoundError("Nici un fișier CSV!")

print(f"Dimensiuni dataset: {df.shape}")
print(df.head())

# ──────────────────────────────────────────────
# 🔧 PRELUCRAREA DATELOR
# ──────────────────────────────────────────────

print("\n" + "="*50)
print("PRELUCRAREA DATELOR")
print("="*50)

df_model = df.copy()

# Parse date
df_model['MatchDate'] = pd.to_datetime(df_model['MatchDate'], errors='coerce')
df_model = df_model.dropna(subset=['MatchDate'])
df_model = df_model.sort_values('MatchDate').reset_index(drop=True)

# Derivăm sezonul (sezonul începe în iulie pentru ligile care încep devreme)
def get_season(date):
    if pd.isna(date):
        return np.nan
    if date.month >= 7:
        return f"{date.year}/{str(date.year + 1)[-2:]}"
    else:
        return f"{date.year - 1}/{str(date.year)[-2:]}"

df_model['Season'] = df_model['MatchDate'].apply(get_season)
df_model['Year']   = df_model['MatchDate'].dt.year
df_model = df_model.dropna(subset=['Season'])

# Ștergem rândurile fără echipe sau rezultat
df_model = df_model.dropna(subset=['HomeTeam', 'AwayTeam', 'FTResult'])

# Codificăm rezultatul
def map_result(result):
    if result == 'H':  return 1
    elif result == 'D': return 0
    elif result == 'A': return -1
    else: return np.nan

df_model['Result'] = df_model['FTResult'].apply(map_result)
df_model = df_model.dropna(subset=['Result'])

print(f"\n📊 Distribuția rezultatelor:")
print(df_model['Result'].value_counts())
print(f"\n📅 Sezoane disponibile: {sorted(df_model['Season'].unique())}")

# ──────────────────────────────────────────────
# ⚙️ INGINERIE DE CARACTERISTICI
# ──────────────────────────────────────────────

print("\n🔨 Pregătim caracteristicile...")

feature_columns = ['HomeElo', 'AwayElo']

form_features      = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
attacking_features = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget', 'HomeCorners', 'AwayCorners']
discipline_features= ['HomeFouls', 'AwayFouls', 'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed']

for feat in form_features + attacking_features + discipline_features:
    if feat in df_model.columns:
        feature_columns.append(feat)

if 'Year' in df_model.columns:
    feature_columns.append('Year')

# Caracteristici derivate
if 'HomeElo' in df_model.columns and 'AwayElo' in df_model.columns:
    df_model['EloDifference'] = df_model['HomeElo'] - df_model['AwayElo']
    df_model['EloTotal']      = df_model['HomeElo'] + df_model['AwayElo']
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

if 'HomeYellow' in df_model.columns and 'HomeRed' in df_model.columns:
    df_model['CardPointsHome'] = df_model['HomeYellow'] + 2 * df_model['HomeRed']
    feature_columns.append('CardPointsHome')

if 'AwayYellow' in df_model.columns and 'AwayRed' in df_model.columns:
    df_model['CardPointsAway'] = df_model['AwayYellow'] + 2 * df_model['AwayRed']
    feature_columns.append('CardPointsAway')

# Umplere valori lipsă
for col in feature_columns:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(0)

print(f"\n✅ Caracteristici selectate ({len(feature_columns)}): {feature_columns}")



# ──────────────────────────────────────────────
# 📈 ROLLING FEATURES (adaugă după sort_values('MatchDate'))
# ──────────────────────────────────────────────

print("\n🔄 Calculez rolling features...")

# Puncte câștigate per meci (3 = victorie, 1 = egal, 0 = înfrângere)
def match_points_home(result):
    if result == 'H': return 3
    elif result == 'D': return 1
    else: return 0

def match_points_away(result):
    if result == 'A': return 3
    elif result == 'D': return 1
    else: return 0

df_model['PointsHome'] = df_model['FTResult'].apply(match_points_home)
df_model['PointsAway'] = df_model['FTResult'].apply(match_points_away)

# ── Construim istoricul per echipă ──
# Pentru fiecare meci, calculăm statistici din ultimele N meciuri ale echipei
# .shift(1) e critic — se asigură că nu folosim datele din meciul curent (data leakage!)

for window in [5, 10]:
    # --- Goluri marcate (media) ---
    # Acasă
    df_model[f'Home_GoalsScored_Last{window}'] = (
        df_model.groupby('HomeTeam')['FTHome']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    # Deplasare
    df_model[f'Away_GoalsScored_Last{window}'] = (
        df_model.groupby('AwayTeam')['FTAway']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # --- Goluri primite (media) ---
    df_model[f'Home_GoalsConceded_Last{window}'] = (
        df_model.groupby('HomeTeam')['FTAway']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_model[f'Away_GoalsConceded_Last{window}'] = (
        df_model.groupby('AwayTeam')['FTHome']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # --- Puncte câștigate (forma) ---
    df_model[f'Home_Points_Last{window}'] = (
        df_model.groupby('HomeTeam')['PointsHome']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_model[f'Away_Points_Last{window}'] = (
        df_model.groupby('AwayTeam')['PointsAway']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# --- Diferențe între echipe (cine e în formă mai bună?) ---
for window in [5, 10]:
    df_model[f'GoalsScored_Diff_Last{window}'] = (
        df_model[f'Home_GoalsScored_Last{window}'] - df_model[f'Away_GoalsScored_Last{window}']
    )
    df_model[f'GoalsConceded_Diff_Last{window}'] = (
        df_model[f'Home_GoalsConceded_Last{window}'] - df_model[f'Away_GoalsConceded_Last{window}']
    )
    df_model[f'Points_Diff_Last{window}'] = (
        df_model[f'Home_Points_Last{window}'] - df_model[f'Away_Points_Last{window}']
    )

# Adaugă toate rolling features în lista de caracteristici
rolling_feature_cols = [col for col in df_model.columns if 'Last5' in col or 'Last10' in col]
for col in rolling_feature_cols:
    df_model[col] = df_model[col].fillna(0)
    if col not in feature_columns:
        feature_columns.append(col)

print(f"✅ Adăugate {len(rolling_feature_cols)} rolling features: {rolling_feature_cols}")


# ──────────────────────────────────────────────
# ✂️ SPLIT CRONOLOGIC PE SEZON
# ──────────────────────────────────────────────

print("\n" + "="*50)
print("SPLIT TRAIN / TEST PE SEZON")
print("="*50)

TEST_SEASON = "2024/25"

train_df = df_model[df_model['Season'] != TEST_SEASON].copy()
test_df  = df_model[df_model['Season'] == TEST_SEASON].copy()

# Verificăm că avem date pentru sezonul de test
if len(test_df) == 0:
    raise ValueError(f"Nu există meciuri pentru sezonul {TEST_SEASON}! Verifică datele.")

X_train = train_df[feature_columns].fillna(0)
y_train = train_df['Result']
X_test  = test_df[feature_columns].fillna(0)
y_test  = test_df['Result']

all_seasons = sorted(train_df['Season'].unique())
print(f"\n📚 Antrenament: {all_seasons[0]} → {all_seasons[-1]}  ({len(X_train)} meciuri)")
print(f"🧪 Test:        {TEST_SEASON}  ({len(X_test)} meciuri)")

# ──────────────────────────────────────────────
# 🤖 ANTRENAREA MODELELOR
# ──────────────────────────────────────────────

print("\n" + "="*50)
print("ANTRENAREA MODELELOR")
print("="*50)

print("\n🌲 Antrenez Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("   ✅ Random Forest antrenat!")

print("\n📖 Antrenez Regresia Logistică...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("   ✅ Regresia Logistică antrenată!")

# XGBoost needs labels as 0, 1, 2 — not -1, 0, 1
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

print("\n⚡ Antrenez XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=2  # keeping it cool
)
xgb_model.fit(X_train, y_train_enc)
print("   ✅ XGBoost antrenat!")

# ──────────────────────────────────────────────
# 🔍 BACKTESTING MECI CU MECI PE 2024/25
# ──────────────────────────────────────────────

print("\n" + "="*50)
print(f"BACKTESTING MECI CU MECI — SEZON {TEST_SEASON}")
print("="*50)
print("\nModelul face predicții folosind statisticile REALE ale fiecărui meci")
print("(Elo, formă, șuturi, cornere etc. din acel meci — fără scor)\n")

result_map = {1: "H", 0: "D", -1: "A"}
label_map  = {1: "Acasă", 0: "Egal", -1: "Deplasare"}

records = []

for idx, row in test_df.iterrows():
    features      = X_test.loc[idx].values.reshape(1, -1)
    actual        = int(row['Result'])

    pred_rf       = int(rf_model.predict(features)[0])
    proba_rf      = rf_model.predict_proba(features)[0]
    pred_lr       = int(lr_model.predict(features)[0])

    correct_rf    = (pred_rf == actual)
    correct_lr    = (pred_lr == actual)

    pred_xgb_enc = int(xgb_model.predict(features)[0])
    pred_xgb = int(le.inverse_transform([pred_xgb_enc])[0])
    correct_xgb = (pred_xgb == actual)

    records.append({
        'Date':         row['MatchDate'].strftime('%Y-%m-%d'),
        'Season':       row['Season'],
        'HomeTeam':     row['HomeTeam'],
        'AwayTeam':     row['AwayTeam'],
        'Score':        f"{int(row['FTHome'])}–{int(row['FTAway'])}",
        'Actual':       label_map[actual],
        'RF_Pred':      label_map[pred_rf],
        'RF_Correct':   '✅' if correct_rf else '❌',
        'LR_Pred':      label_map[pred_lr],
        'LR_Correct':   '✅' if correct_lr else '❌',
        'RF_Conf':      f"{max(proba_rf)*100:.1f}%",
        'actual_int':   actual,
        'rf_pred_int':  pred_rf,
        'lr_pred_int':  pred_lr,
        'XGB_Pred': label_map[pred_xgb],
        'XGB_Correct': '✅' if correct_xgb else '❌',
        'xgb_pred_int': pred_xgb,
    })

results_df = pd.DataFrame(records)

# Afișăm primele 20 meciuri ca preview
print(results_df[['Date','HomeTeam','AwayTeam','Score','Actual',
                   'RF_Pred','RF_Correct','LR_Pred','LR_Correct']].head(20).to_string(index=False))
print(f"\n... și încă {max(0, len(results_df)-20)} meciuri (toate salvate în CSV)")

# ──────────────────────────────────────────────
# 📊 METRICI FINALE
# ──────────────────────────────────────────────

print("\n" + "="*50)
print(f"REZULTATE FINALE — {TEST_SEASON}")
print("="*50)

y_actual  = results_df['actual_int']
y_pred_rf = results_df['rf_pred_int']
y_pred_lr = results_df['lr_pred_int']

acc_rf  = accuracy_score(y_actual, y_pred_rf)
acc_lr  = accuracy_score(y_actual, y_pred_lr)
prec_rf = precision_score(y_actual, y_pred_rf, average='weighted', zero_division=0)
rec_rf  = recall_score(y_actual, y_pred_rf,    average='weighted', zero_division=0)
f1_rf   = f1_score(y_actual, y_pred_rf,        average='weighted', zero_division=0)
prec_lr = precision_score(y_actual, y_pred_lr, average='weighted', zero_division=0)
rec_lr  = recall_score(y_actual, y_pred_lr,    average='weighted', zero_division=0)
f1_lr   = f1_score(y_actual, y_pred_lr,        average='weighted', zero_division=0)

total   = len(results_df)
rf_correct = results_df['RF_Correct'].str.contains('✅').sum()
lr_correct = results_df['LR_Correct'].str.contains('✅').sum()

y_pred_xgb = results_df['xgb_pred_int']
acc_xgb  = accuracy_score(y_actual, y_pred_xgb)
prec_xgb = precision_score(y_actual, y_pred_xgb, average='weighted', zero_division=0)
rec_xgb  = recall_score(y_actual, y_pred_xgb,    average='weighted', zero_division=0)
f1_xgb   = f1_score(y_actual, y_pred_xgb,        average='weighted', zero_division=0)
xgb_correct = results_df['XGB_Correct'].str.contains('✅').sum()

print(f"\n🌲 RANDOM FOREST")
print(f"   Meciuri corecte: {rf_correct} / {total}")
print(f"   Acuratețe:       {acc_rf*100:.2f}%")
print(f"   Precizie:        {prec_rf*100:.2f}%")
print(f"   Sensibilitate:   {rec_rf*100:.2f}%")
print(f"   F1-Score:        {f1_rf*100:.2f}%")

print(f"\n📖 REGRESIE LOGISTICĂ")
print(f"   Meciuri corecte: {lr_correct} / {total}")
print(f"   Acuratețe:       {acc_lr*100:.2f}%")
print(f"   Precizie:        {prec_lr*100:.2f}%")
print(f"   Sensibilitate:   {rec_lr*100:.2f}%")
print(f"   F1-Score:        {f1_lr*100:.2f}%")

print(f"\n⚡ XGBOOST")
print(f"   Meciuri corecte: {xgb_correct} / {total}")
print(f"   Acuratețe:       {acc_xgb*100:.2f}%")
print(f"   Precizie:        {prec_xgb*100:.2f}%")
print(f"   Sensibilitate:   {rec_xgb*100:.2f}%")
print(f"   F1-Score:        {f1_xgb*100:.2f}%")

# Acuratețe per tip de rezultat
print(f"\n📈 ACURATEȚE PER TIP DE REZULTAT (Random Forest):")
for result_val, label in [(-1, 'Deplasare'), (0, 'Egal'), (1, 'Acasă')]:
    mask     = y_actual == result_val
    if mask.sum() == 0:
        continue
    correct  = ((y_actual == result_val) & (y_pred_rf == result_val)).sum()
    total_r  = mask.sum()
    print(f"   {label:<12}: {correct}/{total_r}  ({correct/total_r*100:.1f}%)")

# ──────────────────────────────────────────────
# 💾 SALVARE
# ──────────────────────────────────────────────

print("\n" + "="*50)
print("SALVARE")
print("="*50)

os.makedirs('models', exist_ok=True)
joblib.dump(rf_model,       'models/random_forest_model.pkl')
joblib.dump(lr_model,       'models/logistic_regression_model.pkl')
joblib.dump(feature_columns,'models/feature_columns.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(le,        'models/label_encoder.pkl')
print("✅ Modele salvate în models/")

# CSV cu toate predicțiile pentru sezonul de test
results_df.drop(columns=['actual_int','rf_pred_int','lr_pred_int']) \
          .to_csv('models/backtest_2024_25.csv', index=False)
print("✅ Predicții meci cu meci salvate în models/backtest_2024_25.csv")

# ──────────────────────────────────────────────
# 📊 GRAFICE
# ──────────────────────────────────────────────

print("\n📈 Creez grafice...")
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_actual, y_pred_rf, ax=axes[0],
    display_labels=['Deplasare','Egal','Acasă'])
axes[0].set_title(f"Random Forest — {TEST_SEASON} ({acc_rf*100:.1f}% acuratețe)")

ConfusionMatrixDisplay.from_predictions(y_actual, y_pred_lr, ax=axes[1],
    display_labels=['Deplasare','Egal','Acasă'])
axes[1].set_title(f"Regresie Logistică — {TEST_SEASON} ({acc_lr*100:.1f}% acuratețe)")

plt.tight_layout()
plt.savefig('models/confusion_matrices.png', dpi=100)
print("✅ Matrice de confuzie salvată în models/confusion_matrices.png")
plt.close()

# Importanța caracteristicilor
fig, ax = plt.subplots(figsize=(12, 6))
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]
ax.bar(range(len(importances)), importances[indices])
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_columns[i] for i in indices], rotation=45, ha='right')
ax.set_title("Importanța caracteristicilor (Random Forest)")
ax.set_ylabel("Importanță")
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=100)
print("✅ Importanța caracteristicilor salvată în models/feature_importance.png")
plt.close()

print("\n" + "="*50)
print("🎉 GATA!")
print("="*50)
print(f"\n   Antrenat pe:  {all_seasons[0]} → {all_seasons[-1]}")
print(f"   Testat pe:    {TEST_SEASON}  ({total} meciuri)")
print(f"\n   🌲 Random Forest:       {acc_rf*100:.2f}% acuratețe  ({rf_correct}/{total} corecte)")
print(f"   📖 Regresie Logistică:  {acc_lr*100:.2f}% acuratețe  ({lr_correct}/{total} corecte)")
print(f"\n   Predicții detaliate: models/backtest_2024_25.csv")
print("="*50)