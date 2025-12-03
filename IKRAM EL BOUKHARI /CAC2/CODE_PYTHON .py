# ==============================================================================
# 0. IMPORTATION DES BIBLIOTHÈQUES ET CONFIGURATION
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error, r2_score

# Ignorer les warnings pour un affichage plus propre dans le terminal
import warnings
warnings.filterwarnings("ignore")

# Nom du fichier de données CSV
file_name = "The impact of artificial intelligence on society.csv"

try:
    df = pd.read_csv(file_name)
    print(" Données chargées avec succès.")
except FileNotFoundError:
    print(f"ERREUR: Le fichier '{file_name}' est introuvable. Assurez-vous qu'il est dans le répertoire.")
    exit()

# ==============================================================================
# 1. PRÉ-TRAITEMENT (PREPROCESSING)
# ==============================================================================

print("\n--- Étape 1 : Pré-traitement ---")

# 1.1 Nettoyage des Données (Doublons et ID)
df.drop_duplicates(inplace=True)
df.drop(columns=['ID'], inplace=True, errors='ignore')

# Renommage des colonnes (facilitation du code)
df.columns = [
    'Age', 'Gender', 'Education', 'Employment_Status', 'Occupation', 'Tech_Usage',
    'AI_Knowledge_Self', 'Trust_AI', 'AI_Beneficial_Harmful', 'AI_Usage_Rate',
    'Desire_More_AI', 'AI_Threat_Freedom', 'AI_Eliminate_Professions',
    'Job_Affected_by_AI', 'AI_Ethical_Rules', 'AI_Conscious', 'Q_Not_AI_App',
    'Q_ML_Algorithm', 'Q_ChatGPT_Type'
]

# 1.2 Gestion des Valeurs Manquantes
df['Occupation'].replace('', 'NA_Occupation', inplace=True)
df.replace('I do not know', 'NA_No_Knowledge', inplace=True)
df.replace('I have no idea', 'NA_No_Idea', inplace=True)

for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)


# 1.3 Ingénierie des Caractéristiques et Encodage

# A. Création du Score de Connaissance en IA (Variable Cible)
correct_answers = {
    'Q_Not_AI_App': 'WhatsApp',
    'Q_ML_Algorithm': 'Linear regression',
    'Q_ChatGPT_Type': 'Natural language processing'
}
def create_ai_knowledge_score(row):
    score = 0
    if row['Q_Not_AI_App'] == correct_answers['Q_Not_AI_App']:
        score += 1
    if row['Q_ML_Algorithm'] == correct_answers['Q_ML_Algorithm']:
        score += 1
    if row['Q_ChatGPT_Type'] == correct_answers['Q_ChatGPT_Type']:
        score += 1
    return score
df['AI_Knowledge_Score'] = df.apply(create_ai_knowledge_score, axis=1)

# B. Encodage Ordinal (Likert & Échelles)
# Mapping des niveaux de connaissance déclarés (AI_Knowledge_Self)
knowledge_self_mapping = {'I have no knowledge': 0, "I've heard a little about it": 1, 'I have basic knowledge': 2, 'I have a good level of knowledge': 3, 'I have an expert-level knowledge': 4}
df['AI_Knowledge_Self_Num'] = df['AI_Knowledge_Self'].map(knowledge_self_mapping)

# Mapping standard pour les niveaux d'accord
likert_mapping = {'Strongly disagree': 1, 'I disagree': 2, "I'm undecided": 3, 'Agree': 4, 'Strongly Agree': 5}
agreement_cols = ['AI_Threat_Freedom', 'AI_Ethical_Rules']
for col in agreement_cols:
    df[col + '_Num'] = df[col].map(likert_mapping)

# Autres Mappings Ordinales (Trust, Beneficial, Desire, etc.)
harm_benef_mapping = {'Definitely harmful': 1, 'More harmful than beneficial': 2, 'Both beneficial and harmful': 3, 'More beneficial than harmful': 4, 'Definitely beneficial': 5}
df['AI_Beneficial_Harmful_Num'] = df['AI_Beneficial_Harmful'].map(harm_benef_mapping)

trust_mapping = {"I don't trust it at all": 1, "I don't trust it": 2, "I'm undecided": 3, "I trust it": 4, "I fully trust it": 5}
df['Trust_AI_Num'] = df['Trust_AI'].map(trust_mapping)

desire_mapping = {"Definitely, I would not like to": 1, "I would not like to": 2, "I'm undecided": 3, "I would like to": 4, "Definitely, I would like to": 5}
df['Desire_More_AI_Num'] = df['Desire_More_AI'].map(desire_mapping)

eliminate_mapping = {"Absolutely Can't handle it": 1, "Can't handle it": 2, "NA_No_Idea": 3, "Removes": 4, "Definitely Removes": 5}
df['AI_Eliminate_Professions_Num'] = df['AI_Eliminate_Professions'].map(eliminate_mapping)

job_affected_mapping = {"Definitely I don't think so": 1, "I don't think so": 2, "I'm undecided": 3, "Think": 4, "I definitely think": 5}
df['Job_Affected_by_AI_Num'] = df['Job_Affected_by_AI'].map(job_affected_mapping)

conscious_mapping = {"It certainly can't be": 1, "Can't": 2, "I'm undecided": 3, "Becomes": 4, "Definitely Becomes": 5}
df['AI_Conscious_Num'] = df['AI_Conscious'].map(conscious_mapping)


# C. Encodage One-Hot (Variables Nominales)
nominal_cols = ['Gender', 'Age', 'Education', 'Employment_Status', 'Tech_Usage', 'Occupation']
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)


# D. Création du DataFrame Final (df_final)
cols_to_keep = df_encoded.select_dtypes(include=['int', 'float']).columns.tolist()
df_final = df_encoded[cols_to_keep].copy()
knowledge_qs_cols = ['Q_Not_AI_App', 'Q_ML_Algorithm', 'Q_ChatGPT_Type']
df_final.drop(columns=knowledge_qs_cols + ['AI_Usage_Rate'], errors='ignore', inplace=True)
print(f"Pré-traitement terminé. Nombre de features : {df_final.shape[1]-1}")


# ==============================================================================
# 2. MODÉLISATION ET OPTIMISATION
# ==============================================================================

print("\n--- Étape 2 : Modélisation et Optimisation ---")

# 2.1 Division des Données (Train/Test Split)
y = df_final['AI_Knowledge_Score']
X = df_final.drop(columns=['AI_Knowledge_Score'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2.2 Modèles de Régression à Comparer (Base)
models_to_test = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor (Base)': RandomForestRegressor(random_state=42),
    'KNeighbors Regressor': KNeighborsRegressor()
}
results = {}

for name, model in models_to_test.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2}

print("Comparaison initiale terminée.")


# 2.3 Optimisation du Meilleur Modèle (Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [10, 20, 30, None], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=20, cv=3, verbose=0, random_state=42, 
                               n_jobs=-1, scoring='neg_mean_squared_error')

rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

# Évaluation du modèle optimisé
y_pred_tuned = best_rf.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2_tuned = r2_score(y_test, y_pred_tuned)

results['Random Forest Regressor (Optimisé)'] = {'RMSE': rmse_tuned, 'R2': r2_tuned}
comparison_df_final = pd.DataFrame(results).T.sort_values(by='RMSE')

print(" Optimisation terminée.")
print("\n### Tableau de Synthèse Final des Modèles ###")
print(comparison_df_final.to_markdown(floatfmt=".4f"))


# 2.4 Feature Importance (Importance des Variables)
print("\n--- Étape 2.4 : Importance des Variables (Modèle Final) ---")
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top_10_features = feature_importances.nlargest(10)

print("### Top 10 des Features Influant le AI_Knowledge_Score ###")
print(top_10_features.to_markdown(floatfmt=".4f"))

# ==============================================================================
# FIN DE L'ANALYSE
# ==============================================================================
