# Analyse de Sentiment sur Avis Produits (Amazon)


## Contexte

Les plateformes e-commerce reçoivent des milliers d'avis clients qu'il est impossible d'analyser manuellement à grande échelle. Ce projet s'attaque à ce problème en construisant un classificateur capable de déterminer automatiquement la tonalité (positive ou négative) d'un avis, à partir d'un jeu de données réel : des avis écrits par des clients Amazon sur des produits alimentaires, initialement notés sur une échelle de 5 étoiles et ramenés à un label binaire (+1 pour positif, -1 pour négatif).

## Objectif

Implémenter, entraîner et comparer plusieurs algorithmes de classification linéaire **from scratch** (sans bibliothèque de ML haut niveau), afin de comprendre en profondeur leur fonctionnement interne :
- Perceptron
- Average Perceptron
- Pegasos (SVM à sous-gradient stochastique)

Le projet couvre l'ensemble du pipeline : extraction de features (Bag-of-Words), entraînement, réglage des hyperparamètres, évaluation, et interprétation du modèle.

## Méthodologie

1. **Feature engineering** : transformation des avis textuels en vecteurs Bag-of-Words (indicateurs binaires de présence de mots).
2. **Implémentation des algorithmes** : codage manuel des règles de mise à jour du Perceptron, de l'Average Perceptron et de Pegasos en Python/NumPy.
3. **Réglage des hyperparamètres** : recherche du nombre optimal d'itérations T (parmi [1, 5, 10, 15, 25, 50]) et du paramètre de régularisation λ pour Pegasos (parmi [0.001, 0.01, 0.1, 1, 10]), par évaluation sur un jeu de validation.
4. **Amélioration du dictionnaire** : test de l'impact de la suppression des mots vides (stopwords) sur la performance du modèle.
5. **Interprétabilité** : extraction des mots ayant le plus contribué à la classification positive.

## Résultats

| Algorithme | Précision (validation, T=10) |
|---|---|
| Perceptron | 71,6 % |
| Average Perceptron | 79,8 % |
| Pegasos (λ=0,01) | 79,0 % |

Après réglage optimal des hyperparamètres (T=25) :

| Algorithme | Meilleure précision (validation) |
|---|---|
| Perceptron | 79,4 % |
| Average Perceptron | 80,0 % |
| Pegasos (T=25, λ=0,01) | **80,6 %** |

- **Précision sur le jeu de test (Pegasos, dictionnaire original)** : 80,2 %
- **Précision sur le jeu de test (dictionnaire sans stopwords)** : **80,8 %**
- Précision avec dictionnaire sans stopwords + features de comptage : 77,0 % (moins performant que les indicateurs binaires)

**Top mots les plus discriminants pour la classification positive** : *delicious, great, !, best, perfect, loves, wonderful, glad, love, quickly*

## Stack technique

Python, NumPy

## Ce que ce projet démontre

- Compréhension fine des mécanismes internes des classificateurs linéaires (au-delà de l'utilisation de bibliothèques comme scikit-learn)
- Rigueur méthodologique dans le réglage d'hyperparamètres (validation croisée)
- Capacité à interpréter un modèle de NLP au-delà de la simple métrique de précision
