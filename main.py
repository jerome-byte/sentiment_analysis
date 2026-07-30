import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))



#bag_of_words needs to first be implemented in project 1.py
dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary,  binarize=True)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary,  binarize=True)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
# --- TEST DE VÉRIFICATION ---
print("Valeur maximale dans la matrice d'entraînement :", train_bow_features.max())
if train_bow_features.max() > 1:
    print("ALERTE : La binarisation n'est pas appliquée (valeurs > 1 trouvées).")
else:
    print("Vérification OK : La matrice est binarisée.")
#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')
#
T = 10
L = 0.2
#
thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)
#
def plot_toy_results(algo_name, thetas):
     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
     print('theta_0 for', algo_name, 'is', str(thetas[1]))
     utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)
#
plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------

T = 10
L = 0.01
#
pct_train_accuracy, pct_val_accuracy = \
   p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
#
avg_pct_train_accuracy, avg_pct_val_accuracy = \
    p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
#
avg_peg_train_accuracy, avg_peg_val_accuracy = \
    p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------

data = (train_bow_features, train_labels, val_bow_features, val_labels)
#
# # values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]
#
pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))
#
avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))
#
# # fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))
#
fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))
#
utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------


# Votre code ici
T_best = 25
L_best = 0.01

# Entraînement du meilleur modèle (Pegasos) sur les données d'entraînement complètes
best_theta, best_theta_0 = p1.pegasos(train_bow_features, train_labels, T_best, L_best)

# Prédictions sur l'ensemble de test
test_preds = p1.classify(test_bow_features, best_theta, best_theta_0)

# Calcul de la précision sur l'ensemble de test
test_accuracy = p1.accuracy(test_preds, test_labels)

print("Précision sur l'ensemble de test :", test_accuracy)

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

# On utilise les meilleurs hyperparamètres trouvés à la question précédente
best_theta, best_theta_0 = p1.pegasos(train_bow_features, train_labels, T=25, L=0.01)

wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])

# BONUS : Pour les étiquettes négatives, on regarde les poids les plus faibles (les plus négatifs)
print("\nMost Negative Word Features")
print(sorted_word_features[-10:][::-1])


#-------------------------------------------------------------------------------
# Problem 9 : Suppression des mots vides
#-------------------------------------------------------------------------------

# 1. Créer un nouveau dictionnaire sans les mots vides
dictionary_no_stopwords = p1.bag_of_words(train_texts, remove_stopword=True)

# 2. Extraire les caractéristiques avec ce nouveau dictionnaire
train_bow_features_ns = p1.extract_bow_feature_vectors(train_texts, dictionary_no_stopwords, binarize=True)
val_bow_features_ns = p1.extract_bow_feature_vectors(val_texts, dictionary_no_stopwords, binarize=True)
test_bow_features_ns = p1.extract_bow_feature_vectors(test_texts, dictionary_no_stopwords, binarize=True)

# 3. Entraîner Pegasos avec les meilleurs hyperparamètres (T=25, L=0.01)
# Note : Utilisez les valeurs T_best et L_best que vous avez trouvées précédemment
T_best = 25
L_best = 0.01

best_theta_ns, best_theta_0_ns = p1.pegasos(train_bow_features_ns, train_labels, T_best, L_best)

# 4. Prédire et calculer la précision sur l'ensemble de test
test_preds_ns = p1.classify(test_bow_features_ns, best_theta_ns, best_theta_0_ns)
test_accuracy_ns = p1.accuracy(test_preds_ns, test_labels)

print("Précision sur l'ensemble de test en utilisant le dictionnaire avec les mots vides supprimés :", test_accuracy_ns)


#-------------------------------------------------------------------------------
# Problem 10 : Caractéristiques de comptage (binarize=False)
#-------------------------------------------------------------------------------

# 1. Extraire les caractéristiques avec binarize=False (donc basées sur le comptage)
train_bow_features_count = p1.extract_bow_feature_vectors(train_texts, dictionary_no_stopwords, binarize=False)
test_bow_features_count = p1.extract_bow_feature_vectors(test_texts, dictionary_no_stopwords, binarize=False)

# 2. Entraîner Pegasos avec les mêmes hyperparamètres
best_theta_count, best_theta_0_count = p1.pegasos(train_bow_features_count, train_labels, T_best, L_best)

# 3. Prédire et calculer la précision sur l'ensemble de test
test_preds_count = p1.classify(test_bow_features_count, best_theta_count, best_theta_0_count)
test_accuracy_count = p1.accuracy(test_preds_count, test_labels)

print("Précision sur l'ensemble de test (suppression mots vides + comptage) :", test_accuracy_count)