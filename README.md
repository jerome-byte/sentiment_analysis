
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

Conclusion

La comparaison des trois algorithmes met en évidence des différences claires de comportement et de performance :

Le Perceptron simple est l'algorithme le plus rapide à implémenter et à entraîner, mais aussi le moins performant (71,6 % en validation à T=10, 79,4 % au mieux après réglage). Comme il ne fait que corriger les erreurs une à une sans mémoire des mises à jour passées, ses poids finaux sont sensibles à l'ordre de présentation des exemples et convergent moins bien vers une frontière de décision stable.
Le Perceptron moyen apporte un gain net de performance (79,8 % dès T=10, 80,0 % après réglage) simplement en moyennant les poids sur toutes les itérations. Cette moyenne lisse les oscillations du Perceptron classique et produit une frontière de décision plus robuste, pour un coût de calcul quasi identique — c'est le meilleur compromis simplicité/performance des trois.
Pegasos obtient la meilleure précision globale après réglage des hyperparamètres (80,6 % en validation, 80,2–80,8 % en test selon le dictionnaire utilisé). Grâce à sa régularisation L2 explicite (paramètre λ), il contrôle mieux le compromis biais-variance et généralise légèrement mieux que les deux autres méthodes, au prix d'un réglage plus fin (deux hyperparamètres à optimiser : T et λ).

Le réglage des hyperparamètres s'avère déterminant : les trois algorithmes gagnent entre 0,2 et 8 points de précision entre leur configuration par défaut (T=10) et leur meilleure configuration (T=25). Cela confirme qu'un modèle linéaire, même simple, ne peut être évalué correctement sans une recherche d'hyperparamètres sur un jeu de validation dédié.

Le travail sur les caractéristiques (features) a un impact du même ordre de grandeur que le choix de l'algorithme : retirer les mots vides améliore la précision de test de 0,6 point, tandis que passer d'indicateurs binaires à des comptages de mots la dégrade de plus de 3 points. Cela illustre un principe clé du NLP appliqué à des textes courts comme des avis clients : la présence d'un mot est souvent plus informative que sa fréquence, et un feature engineering bien pensé peut avoir autant d'impact que le choix du classificateur lui-même.

Enfin, l'analyse des mots les plus discriminants (delicious, great, best, perfect, love...) confirme que le modèle a appris une représentation cohérente avec l'intuition humaine du langage positif, ce qui valide la pertinence de l'approche Bag-of-Words malgré sa simplicité.

En résumé, ce projet démontre que même des classificateurs linéaires simples, bien réglés et associés à un feature engineering soigné, permettent d'atteindre une précision d'environ 80 % sur une tâche réelle d'analyse de sentiment — et que Pegasos, grâce à sa régularisation, constitue le meilleur choix parmi les trois lorsque la performance de généralisation prime.


## Installation
- git clone <lien-du-repo>
- cd <nom-du-projet>
- pip install -r requirements.txt

## Utilisation
- python project1/main.py

Le script entraîne les trois algorithmes, effectue le réglage des hyperparamètres sur le jeu de validation, évalue sur le jeu de test, et affiche les mots les plus discriminants.
