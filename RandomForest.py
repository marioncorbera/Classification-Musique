#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:38:50 2022

@author: niels&floriane
"""
#Librairy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import random as rd
import matplotlib.pyplot as plt
import pandas as pd

#======
#    Fonctions de traitement
#======

def load_data(dictionnaire):
    X = np.array(dictionnaire['MFCCs'])
    y = np.array(dictionnaire['label'])
    return X, y

def reformatage(array):
    """
    on reformate les 5 colonnes de 0,1 en 1 colonnes de [0,1,2,3,4]
    qui correspondent au bon genre
    """
    newpred = []
    for k in range (len(array)):
        c = []
        index_max = np.argmax(array[k])
        value_max = max(array[k])
        c.append (index_max)
        for i in range (len(array[k])):
            if array[k][i] == value_max and i != index_max :
                c.append (i)
        newpred.append(rd.choice(c))
    return newpred

def cleaning_0 (array):
    '''
    On veut enlever les lignes de 0 (pas de prédictions) dans le y_pred et donc
    dans y_test pour pouvoir les comparer
    '''
    n = len(array)
    new_array = []
    y_test_new = []
    for k in range(n):
        if sum(array[k]) != 0 :
            new_array.append(array[k])
            y_test_new.append(y_test[k])
    return np.array(new_array), np.array(y_test_new)

'''
Importation et traitement des données
-les données brutes ont déjà été prétraitées
'''
# load data
    #X, y = load_data(DATA_PATH)

X, y = load_data(data)


#on reshape # WARNING : NON ON RESHAPE PAS

Encoder = OneHotEncoder(handle_unknown='ignore')
#y.reshape(len(y),1)
#y = Encoder.fit_transform(y.reshape(len(y),1)).toarray()

#flattening du X
X = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

# create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

'''
Méthode RANDOM FORREST
'''
# Random forrest et fitting du modèle
rf_5 = RandomForestClassifier(n_estimators = 5, random_state = 0)
rf_5.fit(X_train, y_train)


rf_50 = RandomForestClassifier(n_estimators = 50, criterion= "gini")
rf_50.fit(X_train,y_train)

predictions = rf_50.predict(X_test)

#on clean les predictions en enlevant les lignes de 0 et on 
#enlève les lignes équivalentes dans y_test
# y_pred_clean, y_test_clean = cleaning_0(predictions)
# y_pred_reformat = np.array(reformatage(y_pred_clean))
# y_test_reformat = np.array(reformatage(y_test_clean))

#REMARQUE : on en reformate pas y de base donc pas besoin de reformater ici

#Pop/Rock/Rap/Techno/Classique
conf = confusion_matrix(y_test, predictions)
pd.DataFrame(conf, index = ["Pop_données", "Rock_données", "Rap", "Techno", "Classique"],
             columns = ["Pop_pred", "Rock_pred", "Rap", "Techno", "Classique"])

#np.sum(conf,axis = 0)

### Optimisation du nombre d'estimateurs du rf
rfc = RandomForestClassifier(random_state = 42)
param_grid_RF = { 
    'n_estimators': [50, 75, 100],
    'max_depth' : [4,5,6,7,8]
}
#tester aussi mean_split = nb minimum d'individus pour faire une branche

CV_rfc = model_selection.GridSearchCV(estimator = rfc, param_grid = param_grid_RF, cv= 5)
CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
#meilleurs paramètres : {'max_depth': 8, 'n_estimators': 50} 
# pour 0.353 d'accuracy
print("Résultats de la validation croisée :")
for mean, std, params in zip(
        CV_rfc.cv_results_['mean_test_score'], # score moyen
        CV_rfc.cv_results_['std_test_score'],  # écart-type du score
        CV_rfc.cv_results_['params']           # valeur de l'hyperparamètre
    ):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(
        score,
        mean,
        std*2,
        params
    ) )

# model = RandomForestClassifier(random_state=42)
# kfold = RepeatedKFold(n_splits=5,n_repeats=5 ,random_state=42)

# scores = cv_results = cross_val_score(model, X_train, y_train , cv=kfold, verbose=10)
        
# print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

'''
Méthode KNN
'''

from sklearn.neighbors import KNeighborsClassifier

#instanciation et définition du k
knn = KNeighborsClassifier(n_neighbors = 3)
#training
knn.fit(X_train,y_train)

#permet de donner le score de bonne prédiction
knn.score(X_test,y_test)

#correspondance des genres
list_corresp = dict(zip([0,1,2,3,4], ["Pop", "Rock", "Rap", "Techno", "Classique"]))
#partie prédiction

prediction_knn = knn.predict(X_test)

#matrice de confusion

conf = confusion_matrix(y_test, prediction_knn)
conf

#Validation croisée pour trouver les n voisins

from sklearn import neighbors, metrics
from sklearn import model_selection

# Fixer les valeurs des hyperparamètres à tester
#param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}

param_grid = {'n_neighbors':[2,3,4,5]}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy'

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring=score   # score à optimiser
)

# Optimiser ce classifieur sur le jeu d'entraînement
clf.fit(X_train, y_train)

# Afficher le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)

#Résultats : n_ neighbors optimaux = 3

# Afficher les performances correspondantes
print("Résultats de la validation croisée :")
for mean, std, params in zip(
        clf.cv_results_['mean_test_score'], # score moyen
        clf.cv_results_['std_test_score'],  # écart-type du score
        clf.cv_results_['params']           # valeur de l'hyperparamètre
    ):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(
        score,
        mean,
        std*2,
        params
    ) )

