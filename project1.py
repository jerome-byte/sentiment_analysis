from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    L=max(0,1-y(theta.x + theta_0))
    y est le label (+1 ou -1).
    x est le feature_vector.
    theta est le vecteur de poids (theta).
    theta_0 est le biais ou l'offset (theta_0).
    """
    """
        Calcule la perte à charnière pour un point de données unique.
        """
    # 1. Calcul du score brut (produit scalaire de theta et feature_vector + theta_0)
    # np.dot effectue le produit scalaire entre les deux vecteurs.
    score = np.dot(theta, feature_vector) + theta_0

    # 2. Calcul de la valeur à l'intérieur de la fonction max: (1 - label * score)
    # Si le produit (label * score) est >= 1, le modèle est confiant et correct (perte = 0).
    loss_value = 1 - (label * score)

    # 3. Application de la fonction max(0, x)
    # On utilise np.maximum pour comparer la valeur calculée avec 0.
    hinge_loss = np.maximum(0, loss_value)

    return float(hinge_loss)


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Calcule la perte Hinge moyenne sur tout l'ensemble de données.
    """
    # 1. Initialiser une liste pour stocker la perte de chaque exemple
    losses = []

    # 2. Parcourir chaque ligne (chaque point de données) de la matrice
    # feature_matrix est une matrice où chaque ligne est un vecteur de caractéristiques
    for i in range(len(labels)):
        # On extrait le vecteur de caractéristiques de la ligne i
        feature_vector = feature_matrix[i]
        # On extrait le label correspondant à la ligne i
        label = labels[i]

        # 3. Calculer la perte pour ce point spécifique en utilisant votre fonction précédente
        loss = hinge_loss_single(feature_vector, label, theta, theta_0)

        # 4. Ajouter la perte calculée à notre liste
        losses.append(loss)

    # 5. Retourner la moyenne des pertes (somme des pertes / nombre total d'exemples)
    return float(np.mean(losses))

def hinge_loss_full_fast(feature_matrix, labels, theta, theta_0):
    # Calcul direct par produit matriciel : (N, D) dot (D,) -> (N,)
    scores = np.dot(feature_matrix, theta) + theta_0
    # Calcul de la perte pour tous les points d'un coup
    losses = np.maximum(0, 1 - labels * scores)
    # Retourne la moyenne
    return float(np.mean(losses))


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    # 1. Calcul du score pour vérifier si une erreur a été commise
    # Le produit scalaire donne la valeur de décision avant d'ajouter le biais
    score = np.dot(current_theta, feature_vector) + current_theta_0

    # 2. Vérification de l'erreur
    # On utilise une petite valeur epsilon (1e-6) pour gérer les instabilités numériques
    # L'erreur survient si (label * score) est inférieur ou égal à 0.
    if label * score <= 1e-6:
        # 3. Mise à jour des paramètres en cas d'erreur
        new_theta = current_theta + (label * feature_vector)
        new_theta_0 = current_theta_0 + label
    else:
        # 4. Si pas d'erreur, les paramètres restent inchangés
        new_theta = current_theta
        new_theta_0 = current_theta_0

    # 5. Retourne les nouveaux paramètres sous forme de tuple
    return (new_theta, float(new_theta_0))


def perceptron(feature_matrix, labels, T):
    # Récupération du nombre d'échantillons (lignes) et de caractéristiques (colonnes)
    nsamples, nfeatures = feature_matrix.shape

    # Initialisation de theta en tant que vecteur 1D de zéros (taille nfeatures)
    theta = np.zeros(nfeatures)
    # Initialisation de theta_0 (biais) à zéro
    theta_0 = 0.0

    # Boucle sur le nombre total d'itérations T
    for t in range(T):
        # Utilisation de l'ordre spécifié pour le parcours des données
        for i in get_order(nsamples):
            # Extraction du vecteur de caractéristiques et du label correspondant
            feature_vector = feature_matrix[i]
            label = labels[i]

            # Mise à jour des paramètres via la fonction que vous avez créée précédemment
            # On réassigne theta et theta_0 avec les nouvelles valeurs retournées
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)

    # Retourne les paramètres finaux sous forme de tuple
    return (theta, float(theta_0))


def average_perceptron(feature_matrix, labels, T):
    # 1. Initialisation des paramètres de travail
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros(nfeatures)  # Poids actuels
    theta_0 = 0.0  # Biais actuel

    # 2. Initialisation des sommes cumulatives pour calculer la moyenne plus tard
    sum_theta = np.zeros(nfeatures)
    sum_theta_0 = 0.0

    # 3. Boucle principale sur les T itérations
    for t in range(T):
        for i in get_order(nsamples):
            # Mise à jour des poids courants via le Perceptron standard
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)

            # 4. Accumulation des valeurs actuelles dans nos variables de somme
            sum_theta += theta
            sum_theta_0 += theta_0

    # 5. Calcul des moyennes finales
    # Nombre total de mises à jour = T * nombre d'échantillons
    total_updates = T * nsamples
    avg_theta = sum_theta / total_updates
    avg_theta_0 = sum_theta_0 / total_updates

    # Retourne le tuple contenant les paramètres moyennés
    return (avg_theta, float(avg_theta_0))



def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    # 1. Calcul du score (prédiction linéaire)
    # On calcule la valeur de décision actuelle du classificateur
    score = np.dot(theta, feature_vector) + theta_0

    # 2. Vérification de la condition de marge
    # Si le produit (label * score) est <= 1, l'exemple est soit mal classé,
    # soit dans la marge, donc on doit appliquer la mise à jour complète.
    if label * score <= 1:
        # Mise à jour de theta avec régularisation et gradient de perte
        # (1 - eta * L) * theta est la partie régularisation
        # eta * label * feature_vector est le pas de descente de gradient
        new_theta = (1 - eta * L) * theta + (eta * label * feature_vector)
        # Mise à jour du biais (pas de régularisation sur le biais)
        new_theta_0 = theta_0 + (eta * label)
    else:
        # 3. Sinon, on applique uniquement la régularisation (shrinkage)
        # On réduit légèrement le poids theta pour éviter le surapprentissage
        new_theta = (1 - eta * L) * theta
        # Le biais reste inchangé car il n'est pas régularisé
        new_theta_0 = theta_0

    # 4. Retourne les nouveaux paramètres sous forme de tuple
    return (new_theta, float(new_theta_0))


def pegasos(feature_matrix, labels, T, L):
    # 1. Récupération des dimensions de la matrice
    nsamples, nfeatures = feature_matrix.shape

    # 2. Initialisation des paramètres theta (vecteur) et theta_0 (biais)
    theta = np.zeros(nfeatures)
    theta_0 = 0.0

    # 3. Compteur global pour le taux d'apprentissage (t commence à 1)
    count = 1

    # 4. Boucle sur les T itérations
    for t in range(T):
        # Utilisation de l'ordre spécifié par l'exercice pour parcourir les données
        for i in get_order(nsamples):
            # 5. Calcul du taux d'apprentissage dynamique : 1 / sqrt(t)
            # t est le nombre total de mises à jour effectuées (count)
            eta = 1.0 / np.sqrt(count)

            # 6. Extraction des données pour cet échantillon
            feature_vector = feature_matrix[i]
            label = labels[i]

            # 7. Appel de la fonction de mise à jour (Pegasos)
            # On utilise la fonction précédemment implémentée
            theta, theta_0 = pegasos_single_step_update(
                feature_vector, label, L, eta, theta, theta_0
            )

            # 8. Incrémentation du compteur de mises à jour
            count += 1

    # 9. Retourne le tuple final
    return (theta, float(theta_0))



#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    # 1. Calculer les prédictions brutes (scores) pour chaque point de données.
    # On utilise le produit matriciel (@) de feature_matrix par theta, 
    # puis on ajoute le vecteur theta_0 (broadcasté sur toutes les lignes).
    predictions = feature_matrix @ theta + theta_0
    
    # 2. Créer un tableau de résultats initialisé à -1.
    # np.ones_like crée un tableau de la même forme que les prédictions, rempli de 1.
    results = -np.ones_like(predictions)
    
    # 3. Appliquer la condition de classification :
    # Si la prédiction est strictement supérieure à 0, on remplace la valeur par 1.
    # Cela correspond à la règle : "If a prediction is greater than zero, 
    # it should be considered a positive classification."
    results[predictions > 0] = 1
    
    # 4. Retourner le tableau final contenant uniquement 1 et -1.
    return results


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    # 1. Entraîner le classifieur : 
    # On récupère theta et theta_0 en appelant la fonction 'classifier' 
    # avec les données d'entraînement et les arguments optionnels (**kwargs).
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    
    # 2. Obtenir les prédictions :
    # On utilise la fonction 'classify' (implémentée précédemment) sur les deux jeux de données.
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_preds = classify(val_feature_matrix, theta, theta_0)
    
    # 3. Calculer les précisions (accuracy) :
    # On utilise la fonction 'accuracy' fournie pour comparer les prédictions aux labels réels.
    train_acc = accuracy(train_preds, train_labels)
    val_acc = accuracy(val_preds, val_labels)
    
    # 4. Retourner le résultat sous forme de tuple.
    return (train_acc, val_acc)



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):
    indices_by_word = {}  # maps word to unique index
    
    # Charger les mots vides si demandé
    stop_words = set()
    if remove_stopword:
        try:
            with open('stopwords.txt') as f:
                for line in f:
                    stop_words.add(line.strip())
        except FileNotFoundError:
            print("Avertissement : stopwords.txt introuvable.")

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stop_words: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        # On applique un masque booléen : 
            # Si feature_matrix[i, j] > 0, on le met à 1.
            # Cela transforme le comptage de fréquences en indicateur binaire.
        feature_matrix[feature_matrix > 0] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
