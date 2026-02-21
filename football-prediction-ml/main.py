# 🏟️ Prognoza meciuri de fotbal - Cod mai plin de viață decât un meci la penalty! 🤦

import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')  # Ignora avertismentele - nu vrem negativitate!

# 📥 Descarcă datele din Kaggle - la fel ca un gol de zile mari!
print("Descarcă datele meciurilor de fotbal din Kaggle...")
path = kagglehub.dataset_download("adamgbor/club-football-match-data-2000-2025")
print(f"Datele au sosit în: {path}")

# 📂 Afișează ce fișiere sunt în pachet - să vedem ce am primit!
data_files = os.listdir(path)
print(f"\nFișiere disponibile: {data_files}")

# 🏆 Încarcă doar meciurile (nu ratinguri Elo plictisitoare)
matches_path = os.path.join(path, 'Matches.csv')
if os.path.exists(matches_path):
    df = pd.read_csv(matches_path)
    print(f"\n✅ Încarcă fișierul: Matches.csv")
else:
    # 🤔 Planul B: caută orice fișier CSV (optimism!)  
    csv_files = [f for f in data_files if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        print(f"\n✅ Am încărcat: {csv_files[0]}")
    else:
        raise FileNotFoundError("Nici un fișier CSV! Unde e datele, dom'le?!")

print(f"Dimensiuni dataset: {df.shape}  # (rânduri, coloane)")
print(f"\nPrimele rânduri (ca să vedem cu ce muncim):")
print(df.head())
print(f"\nNumele coloanelor (ca să stim cine e cine):")
print(df.columns.tolist())
print(f"\nTipurile de date (numerele și textele lor):")
print(df.dtypes)
print(f"\nValori lipsă (gauri în date):")
print(df.isnull().sum())

# 🔧 Pregătim datele - ca o echipă înainte de meci!
print("\n" + "="*50)
print("PRELUCRAREA DATELOR")
print("="*50)

# 📋 Facem o copie (nu vrem sa distrugem originalele!)
df_model = df.copy()

# ⏰ Convertim datele în format de dată (să fie mai ușor de tras cu ele)
if 'MatchDate' in df_model.columns:
    df_model['MatchDate'] = pd.to_datetime(df_model['MatchDate'], errors='coerce')
    df_model['Year'] = df_model['MatchDate'].dt.year  # Extragem anul

# 🗑️ Ștergem rândurile goale (meciuri fără echipe? Imposibil!)
df_model = df_model.dropna(subset=['HomeTeam', 'AwayTeam', 'FTResult'])

# 🏟️ Traducem rezultatul meciurilor în numere (H=1 acasă, D=0 egal, A=-1 deplasare)
def map_result(result):
    """Transformă rezultatele în numere - ca la fotbal, dar mai simplu!"""
    if result == 'H':  # Acasă a câștigat!
        return 1
    elif result == 'D':  # Egal - frumos dar nu décis
        return 0
    elif result == 'A':  # Deplasare a câștigat - aia e!
        return -1
    else:
        return np.nan

df_model['Result'] = df_model['FTResult'].apply(map_result)  # Aplicăm pe toate rândurile
df_model = df_model.dropna(subset=['Result'])  # Și ștergem ce nu avem rezultat

print(f"\n📊 Distribuția rezultatelor (cine a câștigat mai mult):")
print(df_model['Result'].value_counts())

# ⚙️ Inginerie de caracteristici - adunăm ingredientele pentru rețeta noastră!
print("\n🔨 Pregătim ingredientele...")

# 📈 Selectăm care caracteristici vor intra în model - alegem ce e important
feature_columns = ['HomeElo', 'AwayElo']  # Începem cu ratingurile Elo

# 💪 Adunăm forma echipelor (cum au jucat recent)
form_features = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
for feat in form_features:
    if feat in df_model.columns:
        feature_columns.append(feat)  # Dacă e disponibil, îl luăm!

# ⚽ Adunăm statisticile de atac (șuturi, colțuri - dur cu mingea!)
attacking_features = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget', 'HomeCorners', 'AwayCorners']
for feat in attacking_features:
    if feat in df_model.columns:
        feature_columns.append(feat)

# 🟨 Adunăm datele disciplinei (cărți galbene, roșii - cine e neaștâmpărat?)
discipline_features = ['HomeFouls', 'AwayFouls', 'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed']
for feat in discipline_features:
    if feat in df_model.columns:
        feature_columns.append(feat)

# 📅 Adunăm și anul (pentru a vedea dacă meciurile evoluează)
if 'Year' in df_model.columns:
    feature_columns.append('Year')

# 🧪 Creăm noi caracteristici din cele existente (alchimie de date!)
print("✨ Creăm caracteristici derivate (combinații inteligente)...")

# 📊 Diferența de Elo (cine e mai puternic?)
if 'HomeElo' in df_model.columns and 'AwayElo' in df_model.columns:
    df_model['EloDifference'] = df_model['HomeElo'] - df_model['AwayElo']  # Cine are avantaj
    df_model['EloTotal'] = df_model['HomeElo'] + df_model['AwayElo']  # Calitatea meciurilor
    feature_columns.extend(['EloDifference', 'EloTotal'])

# 📈 Diferența de formă (cine a jucat mai bine recent?)
if 'Form3Home' in df_model.columns and 'Form3Away' in df_model.columns:
    df_model['Form3Diff'] = df_model['Form3Home'] - df_model['Form3Away']
    feature_columns.append('Form3Diff')

if 'Form5Home' in df_model.columns and 'Form5Away' in df_model.columns:
    df_model['Form5Diff'] = df_model['Form5Home'] - df_model['Form5Away']
    feature_columns.append('Form5Diff')

# ⚽ Diferența de șuturi (cine a tras mai mult?)
if 'HomeShots' in df_model.columns and 'AwayShots' in df_model.columns:
    df_model['ShotsDifference'] = df_model['HomeShots'] - df_model['AwayShots']
    feature_columns.append('ShotsDifference')

# 🚩 Diferența de colțuri (cine a avut mai multe ocazii?)
if 'HomeCorners' in df_model.columns and 'AwayCorners' in df_model.columns:
    df_model['CornersDifference'] = df_model['HomeCorners'] - df_model['AwayCorners']
    feature_columns.append('CornersDifference')

# 🟨 Puncte pentru cărți (galbene = 1, roșii = 2 - cine e mai nervos?)
if 'HomeYellow' in df_model.columns and 'HomeRed' in df_model.columns:
    df_model['CardPointsHome'] = df_model['HomeYellow'] + 2 * df_model['HomeRed']
    feature_columns.append('CardPointsHome')

if 'AwayYellow' in df_model.columns and 'AwayRed' in df_model.columns:
    df_model['CardPointsAway'] = df_model['AwayYellow'] + 2 * df_model['AwayRed']
    feature_columns.append('CardPointsAway')

# 0️⃣ Umplim valorile lipsă cu 0 (joacă sigur și pune zerouri!)
for col in feature_columns:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(0)

# 🧹 Ștergem rândurile cu valori lipsă (trebuie date curate!)
X = df_model[feature_columns].dropna()  # Caracteristicile (ingredientele)
y = df_model.loc[X.index, 'Result']  # Ținta (ce vrem să ghicim)

print(f"\n✅ Am selectat {len(feature_columns)} caracteristici: ")
print(f"   {feature_columns}")
print(f"💾 Dimensiuni: {X.shape[0]} meciuri, {X.shape[1]} caracteristici")
print(f"🏟️ Rezultate: {y.shape[0]} meciuri pentru învățat")

# ✂️ Împărțim datele: 80% pentru antrenament, 20% pentru test (ca la antrenament!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📚 Set de antrenament: {X_train.shape[0]} meciuri (să învețe modelul)")
print(f"🧪 Set de test: {X_test.shape[0]} meciuri (să verificăm că nu a furat răspunsurile!)")

# 🤖 Antrenarea modelelor - e ca profesorul să învețe elevii!
print("\n" + "="*50)
print("ANTRENAREA MODELELOR")
print("="*50)

# 🌲 Random Forest - pădure de copaci de decizie (nu, nu e pentru lemne!)
print("\n🌲 Antrenez pădurea aleatorie (100 copaci de gânduri)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Învață din meciuri!
print("   ✅ Pădurea e antrenată!")

# 📖 Regresie Logistică - geometria simplă (cu dreapta o rezolvi pe toată?)
print("\n📖 Antrenez regresie logistică (mai simplă, dar încă deșteaptă)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)  # Și asta învață!
print("   ✅ Regresie antrenată!")

# 📊 Evaluarea modelelor - cum îi merge copiilor noștri de IA?
print("\n" + "="*50)
print("TESTAREA MODELELOR")
print("="*50)

# 🌲 Ce spune pădurea despre testele noi?
y_pred_rf = rf_model.predict(X_test)
print("\n🌲 Rezultate RANDOM FOREST (pădurea):")
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
print(f"   Acuratețe: {acc_rf:.4f}  (de câte ori are dreptate - 0 = prost, 1 = geniu)")
print(f"   Precizie: {prec_rf:.4f}  (când spune ceva, câtde adevărat e?)")
print(f"   Sensibilitate: {rec_rf:.4f}  (vede el toate matchurile importante?)")
print(f"   F1-Score: {f1_rf:.4f}  (balanța perfectă!)")

# 📖 Ce zice linia noastră geometrică?
y_pred_lr = lr_model.predict(X_test)
print("\n📖 Rezultate REGRESIE LOGISTICĂ (linia dreaptă):")
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
print(f"   Acuratețe: {acc_lr:.4f}  (mai simplu, dar funcționează?)")
print(f"   Precizie: {prec_lr:.4f}  (când dă cu piciorul în poartă?)")
print(f"   Sensibilitate: {rec_lr:.4f}  (nu ratează ce e important?)")
print(f"   F1-Score: {f1_lr:.4f}  (cât de bine balanseaza?)")

# 💾 Salvez modelele - ca să nu pierd munca!
print("\n" + "="*50)
print("SALVEZ MODELELE (BACKUP TIME!)")
print("="*50)

os.makedirs('models', exist_ok=True)  # Creaza folder dacă nu există
joblib.dump(rf_model, 'models/random_forest_model.pkl')  # Pădurea în fiola
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')  # Linia în fiola
joblib.dump(feature_columns, 'models/feature_columns.pkl')  # Ingredientele în fiola

print("✅ Pădurea salvată în models/random_forest_model.pkl")
print("✅ Linia geometrică salvată în models/logistic_regression_model.pkl")
print("✅ Ingredientele salvate (ca să nu uităm ce am folosit!)")

# 📊 Desenez grafice - imagini cu rezultatele (femeile/bărbații iubesc graficele!)
print("\n📈 Creez grafice frumoase...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 🔄 Matricile de confuzie (cine s-a încurcat cu cine?)
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=axes[0])
axes[0].set_title("Pădurea: Cine a ghicit bine/greșit?")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=axes[1])
axes[1].set_title("Linia: Cine a ghicit bine/greșit?")

plt.tight_layout()
plt.savefig('models/confusion_matrices.png', dpi=100)
print("✅ Matricile de confuzie salvate în models/confusion_matrices.png")
plt.close()

# 🏆 Importanța caracteristicilor (care ingrediente sunt cu adevărat importanți?)
fig, ax = plt.subplots(figsize=(10, 6))
importances = rf_model.feature_importances_  # Ce cred pădurea că e important
indices = np.argsort(importances)[::-1]  # Sortez de la mai important la mai puțin
ax.bar(range(len(importances)), importances[indices])
ax.set_xlabel("Ce caracteristică?")
ax.set_ylabel("Cât de important (ponderea)?")
ax.set_title("Pădurea zice: Ce conteaza VRAIMENT?")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels(feature_columns, rotation=45, ha='right')  # Rotesc labels ca să se citească
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=100)
print("✅ Importanța caracteristicilor salvată în models/feature_importance.png")
plt.close()

print("\n" + "="*50)
print("🎉 PROIECTUL E GATA! 🎉")
print("="*50)
print("\n✨ Modelele sunt antrenate și salvate!")
print(f"   Am folosit {len(feature_columns)} caracteristici inteligente")
print(f"   Pădurea are acuratețe: {acc_rf*100:.2f}%  (cam de câte ori ghicește bine)")
print(f"   Linia are acuratețe: {acc_lr*100:.2f}%  (mai simplă, dar merge!)")
print("\n🚀 Acum poți face predicții!")
print("   Rulează: python predict_match.py \"Manchester United\" \"Liverpool\"")
print("   Și te va spune cine câștigă! (sau nu... nu e adesea sigur în fotbal 😄)")
print("="*50)
