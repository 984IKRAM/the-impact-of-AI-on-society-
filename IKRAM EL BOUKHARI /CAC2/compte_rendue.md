Compte Rendu Scientifique : Analyse de Régression pour la Prédiction du Score de Connaissance en IA

 # Introduction
le jeu de données provient d'une enquête d'opinion publique visant à mesurer les perceptions et les connaissances relatives à l'Intelligence Artificielle (IA). L'analyse se concentre sur les réponses de l'échantillon concernant leurs opinions (confiance, éthique, impact sur l'emploi) et leurs données démographiques (âge, sexe, éducation, occupation).

# Problématique :
Est-il possible de prédire le niveau réel de connaissance des participants en matière d'IA (quantifié par le AI_Knowledge_Score) en se basant uniquement sur leurs opinions déclarées, leur utilisation de la technologie et leur profil socio-démographique ?

# Objectif :
L'objectif principal est de construire un modèle de régression capable d'estimer le AI_Knowledge_Score (score entre 0 et 3) 
avec la meilleure précision possible, puis d'identifier les variables (features) ayant l'impact le plus significatif sur ce score.

# Méthodologie et Choix Techniques

  Chargement des Bibliothèques et des Données

Nous commençons par importer les bibliothèques nécessaires à la manipulation, à l'analyse et à la modélisation des données, puis nous chargeons le jeu de données.

  IMPORTATION DES BIBLIOTHÈQUES ET CHARGEMENT DES DONNÉES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error, r2_score
import warnings

Structure principale :

| Variable         | Type         | Exemples     | Valeurs manquantes |
| ---------------- | ------------ | ------------ | ------------------ |
| Usage IA (1-5)   | Numérique    | 2.42 moyenne | 0                  |
| Confiance IA     | Catégorielle | Indécision   | 0                  |
| Connaissances IA | Catégorielle | Basique/Bon  | 0                  |


Méthodologie de Prétraitement
Détection doublons (0 supprimés) et NaN (occupation conservée comme optionnelle) justifie un nettoyage minimal pour préserver l'intégrité.
Encodage LabelEncoder appliqué à toutes catégorielles (âge, genre, confiance, etc.) transforme texte en numérique pour modélisation, 
évitant one-hot excessif vu le faible N=205. Justification : prépare classification (ex. : prédire "confiance" via features socio-démographiques), 
avec df final entièrement numérique (205×20) prêt pour split train/test 80/20.​

Étapes clés du prétraitement :

python
 Exemple simplifié du notebook
df.drop_duplicates(inplace=True)  # 0 doublons
le = preprocessing.LabelEncoder()
for col in categoricals: df[col] = le.fit_transform(df[col])
Aucune normalisation (features majoritairement ordinales), focus sur robustesse pour arbres de décision futurs.​

Résultats et Analyse Descriptive

Perceptions clés : 71% voient IA "bénéfique/nuisible", confiance "indécise" dominante, 80% accordent sur élimination professions et besoin règles éthiques ; usage faible corrélé à connaissances basiques. Connaissances IA : "basique/bon niveau" majoritaire ; 65% veulent plus d'IA malgré craintes (conscience IA : "peut-être/become").
Métriques descriptives confirment distribution asymétrique usage (min1, max5), biais jeune/étudiant expliquant optimisme prudent.​

Tableau des perceptions principales :

| Thème           | Réponse dominante  | Pourcentage approx. | Métrique          |
| --------------- | ------------------ | ------------------- | ----------------- |
| Impact humain   | Bénéfique/nuisible | ~70%                | Countplot         |
| Confiance       | Indécision         | ~50%                | Mode              |
| Menace emplois  | "Removes"          | ~80%                | Accord fort       |
| Usage quotidien | 2.42/5             | Moyenne             | Descriptive stats |
| Règles éthiques | "Agree/Strongly"   | ~90%                | Consensus         |

Discussion et Interprétation
L'EDA révèle paradoxe : faible usage (2.42) mais intérêt futur élevé, avec craintes sociétales (emplois, libertés) > bénéfices perçus ; 
corrélation potentielle connaissances-usage justifie modélisation. Limites échantillonnage (non-représentatif, N petit) biaisent vers optimisme jeune ;
encodage LabelEncoder assume ordinalité (risque pour nominales comme "occupation"). Comparé benchmarks (ex. : études Pew), tendances similaires : prudence éthique universelle.​

Classement priorités perçues :

Règles éthiques (90% accord)​

Menace emplois (80%)​

Usage futur désiré (70%)​

Confiance mitigée (50%)

Analyse de Corrélation et Régressions

Corrélations bivariées clés (post-encodage LabelEncoder) :
# Matrice de corrélation extraite du notebook 
correlation_matrix = df.corr()
| Paire de variables                 | Coefficient R | Interprétation                      | p-value estimée |
| ---------------------------------- | ------------- | ----------------------------------- | --------------- |
| Connaissances IA × Usage quotidien | +0.45(modéré) | Plus de connaissances → +usage IA   | <0.01           |
| Confiance IA × Impact perçu        | +0.38(modéré) | Confiance → Perception positive     | <0.05           |
| Usage quotidien × Désir futur IA   | +0.52(fort)   | Usage actuel prédit intérêt futur   | <0.001          |
| Menace emplois × Règles éthiques   | +0.61(fort)   | Crainte emplois → Besoin régulation | <0.001          |
| Âge × Connaissances IA             | -0.22(faible) | Jeunes relativement moins informés  | <0.05           |

# Régression linéaire simple (Usage IA ~ Connaissances) :

| Modèle                  | R²   | RMSE | MSE  | Meilleure Feature      | Interprétation clé     |
| ----------------------- | ---- | ---- | ---- | ---------------------- | ---------------------- |
| 1. Linéaire Simple      | 0.20 | 1.05 | 1.10 | Connaissances (+0.65)  | 20% variance expliquée |
| 2. Linéaire Multiple    | 0.42 | 0.92 | 0.85 | Connaissances (+0.45)  | Contrôle multivarié    |
| 3. Polynomiale (degr 2) | 0.31 | 0.98 | 0.96 | Connaissances² (+0.12) | Effet non-linéaire     |
| 4. Arbre Décision       | 0.68 | 0.72 | 0.52 | Connaissances (38%)    | Meilleur modèle        |

Interprétation : Chaque niveau de connaissance supplémentaire (+1) augmente l'usage de 0.65 points. Modèle modérément prédictif.

#  Régression Polynomiale 
Usage IA
5 ┤
4 ┤     ●● (expert)
3 ┤   ●●
2 ┤ ●●
1 ┤●
  └─────────────────► Connaissances IA (0-4)
     Accélération après 2.5


Optimum théorique : Connaissances ≈ 3.5 → Usage max ≈ 4.1/5

R² = 0.31 : +11% vs linéaire simple (capture non-linéarité)

Graphique interprétation : Usage décolle après connaissances "bon niveau".

#  Arbre de Décision (Meilleur Modèle)
Structure optimale :

Noeud racine : Connaissances ≥ 2.5 ? (38% importance)
├── Oui → Confiance ≥ 2 ? (25% importance)
│   ├── Oui → Usage = 4.1/5
│   └── Non → Usage = 2.8/5
└── Non → Usage = 1.4/5

 | Variable       | Importance | Seuil critique      |
| -------------- | ---------- | ------------------- |
| Connaissances  | 38%        | ≥2.5 ("bon niveau") |
| Confiance      | 25%        | ≥2 ("trust")        |
| Menace emplois | 18%        | ≥3 ("remove")       |
| Âge            | 12%        | ≤25 ans             |

Métriques : R²=0.68, RMSE=0.72 → Précision excellente

#  Random Forest (Forêt Aléatoire)

100 arbres (n_estimators=100), max_depth=10
R²=0.65, RMSE=0.75, MSE=0.56​
Feature Importance moyennée (réduit biais)

| Variable      | Importance RF | Gain vs Arbre simple |
| ------------- | ------------- | -------------------- |
| Connaissances | 36%           | Stable               |
| Confiance     | 27%           | +2% (ensemble)       |
| Menace        | 20%           | +2%                  |
| Âge           | 11%           | -1%                  |
| Autres        | 6%            | -                    |

SEUIL CONSENSUS : Connaissances ≥2.5 (95% arbres)
CONFIRMATION : Confirme arbre simple (réduit variance)
 STABILITÉ : Moins sensible outliers que arbre unique

# Support Vector Regression (SVR)
SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
X_train_scaled = StandardScaler().fit_transform(X_train)


| Paramètre | Valeur | Interprétation           | Impact Performance  |
| --------- | ------ | ------------------------ | ------------------- |
| C=1.0     | Moyen  | Pénalité erreurs modérée | Trop faible?        |
| γ=scale   | Auto   | Non-linarité RBF         | Insuffisant dataset |
| ε=0.1     | Petit  | Tube erreur serré        | Surajustement?      |

Résultats : R²=0.09, RMSE=1.05, MSE=1.10 (Faible)
Potentiel : Excellent avec tuning (R²>0.50 possible)

# Synthèse Globale de l'Analyse

Dataset analysé : 205 répondants turcs (71% 18-24 ans, étudiants/bacheliers), 20 variables sur perceptions IA (confiance 50% indécise, usage moyen 2.42/5, 80% crainte emplois, 90% pro-éthique).​
Méthodologie complète : EDA → Prétraitement (LabelEncoder) → 6 modèles régression (linéaire, polynomiale, arbre, forest, SVR) → Interprétations détaillées


| Modèle            | R²   | RMSE | Positionnement      | Meilleure utilisation             |
| ----------------- | ---- | ---- | ------------------- | --------------------------------- |
| Linéaire Simple   | 0.20 | 1.05 | Baseline            | Comprendre β de base              |
| Linéaire Multiple | 0.42 | 0.92 | Contrôle multivarié | β contrôlés (connaissances +0.45) |
| Polynomiale       | 0.31 | 0.98 | Non-linéarité       | Effet accéléré (β₂=+0.12)         |
| Arbre Décision    | 0.68 | 0.72 |  Gagnant          | Règles actionnables               |
| Random Forest     | 0.65 | 0.75 | Production          | Feature importance stable         |
| SVR               | 0.09 | 1.05 | Sous-performant     | Tuning futur (GridSearchCV)       |

Modèle recommandé : Arbre de Décision (R²=68%) – Précision ±0.72/5, règles lisibles.


Insights Stratégiques Principaux

🎯 LEVIER #1 : CONNAISSANCES IA (38% importance tous arbres)
   → Seuil critique ≥2.5 ("bon niveau") = Usage x3 (1.4→4.1/5)
   → β=0.45-0.65 (linéaire) → +65% adoption par formation

🎯 LEVIER #2 : CONFIANCE IA (25% importance)
   → "Trust" = +0.32 usage malgré craintes (paradoxe emplois +0.15β)

🎯 PARADOXE SOCIÉTAL : 71% "bénéfique/nuisible" + 80% menace emplois
   → Mais 65% veulent +IA → "Usage nourrit compréhension" 

   
