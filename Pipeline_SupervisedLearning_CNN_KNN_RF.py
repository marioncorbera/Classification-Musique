# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:35:49 2022
@author: marion.corbera, niels.andre, floriane.cornue"

L'objectif de ce sript est de contsruire un modèle permettant de classifier le
genre musical d'une chanson. 
Nous avons choisi les genres : Classique, Pop, Rap, Rock & Techno de manière 
arbitraire. Nous avons utilisé comme jeu de données 10 chanson/musiques par 
genres, considérées comme 'représentatives' de leur genre, soit 50 musiques au 
total.  

Ce script montre comment nous avous pu prétraiter et entraîné notre mosèle. Il
est composé de deux parties:
    
-   La première partie décrit la "pipeline" de prétraitements appliquée à nos 
    musiques.
-   La deuxième partie comporte l'entraînement et la sélection de 3 types de 
    modèles classifications our classer nos genres musicaux:
    -> Un réseau de neurones convolutif : CNN (packages + fonctions +
    compilation des fonctions).
    -> Un modèle de forêt aléatoire : RF (packages + fonctions + compilation 
    des fonctions).
    -> Un modèle des K plus proches voisins : KNN (packages + fonctions + 
    compilation des fonctions).
    
"""
 
###############################################################################
### PROCESSING PIPELINE 
###############################################################################

###############################################################################
### SCRIPT PREPROCESS :
   
   # 1- Chargement des musiques.
   # 2- Découpages les audio en échnatillons de x secondes.
   # 3- Preprocess échantillons (normalisation, découpage, padding & extraction des features demandées).
   # 4- Stockage des label, nom des séquences, nom des label et des features dans un dictionnaire.
   
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################
  
import os
import librosa
from tqdm import tqdm
import numpy as np

###############################################################################       


###############################################################################       
### CREATION FONCTIONS
###############################################################################


def initialisor(wanted_feature):
    # création d'un dictionnaire vide 
    # zone stockage données & leur noms
    dataset = {
        "mapping": [], # nom des classes
        "name": [], # nom de la séquence
        "label":[], # normalement numerique => encoder pour être réutilisée comme label par le réseau
        str(wanted_feature): [] # le feature qu'ont veut extraire (MFCC's, spectrogramme...)
    }
    
    return dataset



def normaliser(record):
    # normalisation des données MinMax
    # nromaliser par MinMax permet de ne pas écraser les valeurs médianes de l'échantillons par ses plus valeurs extrêmes.
    norm_record = (record - record.min()) / (record.max() - record.min())
    
    return norm_record



def loader(file_path, sr, mono):
    # on load les musiques
    # sr : freq echantillonnage => on garde la ferquence "naturelle" de préférence
    # mono : nb de cannaux
    record = librosa.load(file_path, sr, mono)[0]   
    
    # normalise
    norm_record = normaliser(record)
        
    return norm_record



def padder(sequence, sr, seq_time):
    # ajoute 0 si sequence trop courte => singal vide
    expected_len = sr*seq_time # = nb de point de la séquence attendu
    if len(sequence)<expected_len:
        missing_len = int(expected_len - len(sequence))
        # rajout des 0 en début de séquence
        sequence =  np.pad(sequence, (0, missing_len), mode='constant')

    return sequence
     


def features_extractor(sample, sr, wanted_feature, frame_size, hop_length, n_mfcc):
    # extrait les features demandés
    
    if wanted_feature == 'spectrogramme':
        # spectrogramme = "photo" energie/frequence durant le temps
        # transformée de Fourier permet de passer du domaine du temps à celui de la fréquence
        # stft = short time fourier transform (on ne fait pas la transformer de foruier classique car ce ne sont pas des sons "périodiques".)
        # stft => on fait une fft (fast fourier transform sur chaque frame) 
        # stft permet de passer dans le domaine temps & fréquence!
        stft = librosa.core.stft(sample, frame_size, hop_length) # matrice complexe numbers
        # extraction du spectrogramme de la séquence
        features = np.abs(stft)**2 # on passe au carré pour repasser dans le domaines du réél après la stft
        
    elif wanted_feature == 'log_spectrogramme':
        # short time fourier transform
        stft = librosa.core.stft(sample, frame_size, hop_length) # matrice complexe numbers
        spectrogramme = np.abs(stft)**2        
        features = librosa.power_to_db(spectrogramme)
        
    elif wanted_feature == 'MFCCs':
        # MFCCs = Mel-frequency cepstral coefficients 
        # Combinaison filtre mel spectrogramme & cepstral centroïds 

        # Mel spectrogramme => simule comment l'oreille humaine perçoit le pitch
        # appartient au domaine temps frequence avec ampltiude en log et freq en log
        # mel = 2595.log(1+(freq/500))
        # mel bands => nombre de points mel (donc nb de frequence considérées)
        
        # cepstral centroïds 
        # représentation des frequences où la plupart de l'enregie est concentrée
        
        # MFCCs =>  => utilise échelle "mel" pour réduire la dimentionalité des "sons" avec un filtre triangulaire appliqué au fréquences & cesptrum pour optimiser les fréquences pour lesquelles le son prend le plus d'énergie.
        mfccs = librosa.feature.mfcc(sample, sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        # il faut prendre la transposée pour bien avoir les indices des séquences en ligne!
        features = mfccs.T
        
    elif wanted_feature == 'MFCCs_delta':
        # permet d'avoir aussi une indication sur comment les MFCCs évoluent dans le temps (dérive MFFCs => ordre 1 & 2)
        mfccs = librosa.feature.mfcc(sample, sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length)
        # dérivée d'orde 1
        delta_mfccs = librosa.feature.delta(mfccs)
        # dérivée d'ordre 2
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        # on "empile" les MFCCs et leur déérivées
        d_mfccs = np.concatenate((mfccs,delta_mfccs,delta2_mfccs))
        features = d_mfccs.T
    
    else:
        print('feature demandé non reconnue/codée pour le moment')
    
    
    return features.tolist()
       



def sequenceur(record, sr, seq_time, overlap, wanted_feature, frame_size, hop_length, n_mfcc, dataset, file_name, i):
    # seq_time : durée des sequences en secoondes
    # overlap : chevauchement entre sequence
    
    # durée de la musique entière
    duration = len(record)/sr
    # nombre de segments pour découper la chanson
    nb_segments = int(duration/seq_time)
        
    # boucle pour tout les segments d'un audio
    for seq in tqdm(range(nb_segments+1), desc= "slicing file in sequences"): 
            
        # détermine début séquence 'seq'
        start = int((seq_time*sr-sr*overlap)*seq)
        finish = int(start + seq_time*sr)
            
        # sauvegarde la séquence => les points de début et de fin sur la musique
        sequence = record[start:finish]
        # on récuoère le titre & on ajout seq et le numero de la séquence
        sample_name = file_name+'seq_'+str(seq+1)
        
        # pad si besoin (séquence trop courte)
        sample = padder(sequence, sr, seq_time)
            
        # extrait features demandé
        features = features_extractor(sample, sr, wanted_feature, frame_size, hop_length, n_mfcc)
        
        # mise en place de notre dataset
        dataset["label"].append(i-1) # le numero de label la séquence
        dataset["name"].append(sample_name) # nom de la séquence
        dataset[str(wanted_feature)].append(features) # featrue extrait de la séquence
        
    return dataset           



    
def Pipeline_preprocessing(datapath, sr=22050, mono=True, seq_time=0.5, overlap=0.1, wanted_feature='MFCCs_delta', frame_size=2048, hop_length=512, n_mfcc=13):
   
    # record path = chemin des recodings originaux
    # sr = sample rate
    # mono = nb de channel
    # seq_time = durée sous séquence => echantillon
    # over_lap = chevauchement sequence
    # wanted_feature = feature à extraire (spectro, log_spectro & mfccs...)
    # frame_size = gde frame size : aumengte freq resolution / baisse temps resolution VS petite frame size : baisse freq resolution / aumengtes temps resolution 
    # hop_length = nb de pts on décalle vers droite pour changer de frame
    # n_mfcc = nb mel_band => nb de point mel echelle (en gros nombre de filtres triangulaires dont on se sert)


    # création dataframes stockages
    dataset = initialisor(wanted_feature)
    
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(datapath)): 
        # dirpath = dossier actuel
        # dirnames = sous-dossier
        # filenames = tout les fichier dans dirpath
        pass
    
        # fonction os.walk va mettre dataset_path = dir_path donc on s'assure de bien être au niveau sous-dossiers (dirnames)
        if dirpath is not datapath:
            
            # sauvegarde nom des labels 
            semantic_label = dirpath.split("\\")[-1] # F:\Marion\....\Rock => met dans liste chaque mot => on grade que dernier => 'Rock' par exemple
            print("\nProcessing: {}".format(semantic_label))  
            
            # mapping pour les labels => on garde le nom du genre musique (semantic_label)
            dataset["mapping"].append(semantic_label)
            
        
            for file in tqdm(os.listdir(dirpath), desc="processing files in the current path"): # juste nom du file => il nous faut le file path
                
                # on récupère le file name et on enlève l'extension du fichier
                file_name = file.replace(".WAV", "")
                file_name = file.replace(".wav", "")

                # on load les audio par genre musicaux (= par sous dossiers)
                file_path = os.path.join(dirpath, file)
                record = loader(file_path, sr=sr, mono=mono) 
        
                # normalisation, découpage, padding & extractaction des features
                dataset = sequenceur(record,
                                     sr,
                                     seq_time,
                                     overlap,
                                     wanted_feature,
                                     frame_size,
                                     hop_length,
                                     n_mfcc,
                                     dataset,
                                     file_name,
                                     i)

    
    return dataset

###############################################################################       


###############################################################################       
###  PARAMETRAGE & LANCEMENT DU PIPELINDE DE PREPROCESSING
###############################################################################       

# DATAPATH = le chemin du dossiers contenant les fichiers avec les genre musicaux
# SR = sample rate => on met la frequence d'échantillonage de base des échantillons (pour ne pas perdre en résolution)
# MONO = True car on a que 1 seul cannal
# SEQ_TIME = durée des séuqneces => 4 secondes nous permettent 
# OVER_LAP = durée du chevauchement entre les séquences
# WANTED_FEATURE = le feature que l'ont souhaite extraire dans les séquences
# FRAME_SIZE = taille de la fenêtre (gde frame size : aumengte freq resolution / baisse temps resolution VS petite frame size : baisse freq resolution / aumengtes temps resolution)
# HOP_LENGTH = déplacement dans fenêtre (nb de pts on décalle vers droite pour changer de frame)
# N_MFCCS = nb de point mel utilisée echelle


DATAPATH = r'C:\Users\mario\Desktop\M2_ACO\Machine_learning\DL_projet\music_samples_test'
SR = 22050              
MONO = True
SEQ_TIME = 4
OVER_LAP = 0.05
WANTED_FEATURE = 'MFCCs_delta' 
FRAME_SIZE = 2048  
HOP_LENGTH = 512
N_MFCCS = 13


if __name__ == "__main__":
    data = Pipeline_preprocessing(DATAPATH, sr=SR, wanted_feature=WANTED_FEATURE, n_mfcc=N_MFCCS)    
      
###############################################################################




###############################################################################
### CONVOLUTIONAL NEURAL NERTWORK 
###############################################################################

###############################################################################
### SCRIPT CNN :
   
   # 1- Chargement des données
   # 2- Création des train, test, et validation sets & mise en forme
   # 3- Création de l'architecture du modèle
   # 4- Compilation & entraînenement le modèle
   # 5- Sauvegarde du modèle
   # 6- Comparaison du genre prédit par le CNN & genre réél d'une séquence
   
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################

import tensorflow as tf 
tf.test.is_gpu_available() 

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

###############################################################################


###############################################################################       
### CREATION FONCTIONS
###############################################################################

# simule loadding => en gros pour si on recharge les données. (Voir si on les passe en .json si c'est trop gros)
def load_data(dictionnaire):
    X = np.array(dictionnaire[str(WANTED_FEATURE)])
    y = np.array(dictionnaire['label'])
    return X, y

def plot_history(history):
    # 2 plots l'accuracy & la loss pour les trainset & validationset en fonction des époques

    # on fait le subplots
    fig, axs = plt.subplots(2)

    # subplot 1
    # accuracy sublpot => on check que test_accuracy descend
    # pour python 3.6
    # car je dois être en envrironement python 3.6
    # parce les cartes graphiques Radeon ne marche qu'avec une version rétrogradée tensorflow (donc j'ai une version rétrogrdée de python aussi)
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    
    # pour python 3.9
    #axs[0].plot(history.history["accuracy"], label="train accuracy")
    #axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # subplot 2
    # error sublpot => on check que test_error ne remonte pas (si remonte on a fait trop de répétition d'époques)
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    
    # prépare les JDD (split train, test , validation) => retourne les jdd partitionnés:
    # test_size : pourcentage des données mise dans le testset (ne participe 
    # pas à l'entraînement ni à l'évaluation du modèle => test accuracy du modèle retenue => n'est jamais vu inderectement par le modèle)
    # validation_size : pourcentage des données mise dans le validationset (sert à évaluer le modèle)


    # chargement des données:
    X, y = load_data(data)


    # création des train, validation & test split
    # test_set / trainset (temporaire) => on n'utilisera pas testset avant d'avoir le modèle "final"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # on repartitionne trainset => entrainement modèle / validationset => évaluation du modèle
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # on ajoute un axe a nos input sets => car il faut 3D pour tensorflow et 
    # dimensions pour une sequences (3D) : time_bins * deltaMFCCs * channel => (22*39*1) 
    # 1 = channel (on reste dans le noir et blanc pas nécéssaire de passer RGB)
    X_train = X_train[..., np.newaxis] # 4D array =>  (num_séquence * 22 * 39 * 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    
    # on créée le modèle (structure + hyperparamètres) => retourne un CNN (CNN => bien pour traitement d'image)
    # input_shape : shape de nos données
 
    # va permettre "d'empiler" nos couches les unes après les autres (linéaire)
    model = keras.Sequential()

    # Première couche de convolution
    # model.add => ajout d'une couche
    # 32 filtres (kernels) et grid de (3,3)
    # activation = 'relu' (rectified linear unit) car est efficace dans les problémes de classification multiclase
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # grid size (3,3), stride(2,2) & padding( =same)
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # batchnormalisation => process "normalise" l'activation dans la couche en cours
    # augmente fiabilité & la vitesse à laquelle le modèle va converger
    model.add(keras.layers.BatchNormalization()) # => augmente vitesse apprentissage modèle

    # Deuxième couche de convolution
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Troisième couche de convolution
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # on flatten les outputs des convolitions et on les met dans une couche "dense" (la couche basique sans filtre de convolution ni rien)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu')) 
    # Dropout = modèle plus robuste car étient au hasard neurones pendant le training du modèle (0.35 => frequence du dropout dans la couche)
    model.add(keras.layers.Dropout(0.35)) # limite overfitting

    # output layer => couche de sortie 
    # 5 neurones de sortie de sortie car 5 genres musicaux à prédire
    # "classifieur" softmax => premet sélectionner classe la plus probable 
    model.add(keras.layers.Dense(5, activation='softmax')) 

    return model


def predict(model, X, y):
    
    # fonction pour pédire la classe d'un échantillon
    # model : CNN entrainé & retenu
    # X : échantillon
    # y : label

    # comme tout à l'heure:
    # on ajoute un axe a nos données car il faut 3D pour tensorflow 
    # donc X = 1 * 22 * 39 * 1
    X = X[np.newaxis, ...] 

    # prédit le genre musical le plus prbable
    prediction = model.predict(X)

    # récupère index avec la valeur max => genre le plus probable d'appartenance
    predicted_index = np.argmax(prediction, axis=1) 

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

###############################################################################


###############################################################################       
###  PARAMETRAGE & CREATION DU CNN
###############################################################################       

if __name__ == "__main__":

    # on split les données en train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # création du CNN
    input_shape = (X_train.shape[1], X_train.shape[2], 1) # on prend 22*39*1 (time bins* delta mfcss * channel)
    model = build_model(input_shape)

    # on compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy', # = bien pour les classifications multiclasse (change de logloss)
                  metrics=['accuracy'])

    model.summary()

    # entrainement model
    # on aurait pu tester le batch size mais pas le temps...
    # on stock les répétitions dans l'history (pour les plots & surveiller overfitting)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        # on ajoute un "ealy stopping" pour éviter d'overfitter & tourner dans le vide
                        # si le modèle ne s'est pas "améliorer" pendant k époques, les répétitions de l'entraînement s'arrêtent
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)],
                        batch_size=32,
                        epochs=100) 

    # plot accuracy & error pour le training & validation (on peut voir si le modèle apprend pdt entrainement et surveille overfitting)
    plot_history(history)

    # on évalue l'accuracy du modèle avec le test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)


### Test rapide:
#X_to_predict = X_test[2]
#y_to_predict = y_test[2]

# predict sample => compare label observé et label réél
#predict(model, X_to_predict, y_to_predict) 

###############################################################################



###############################################################################
### RANDOM FOREST
###############################################################################

###############################################################################
### SCRIPT RF :
   
   # 1- On charge le dataset avec la fonction load_data et on reshape le X afin 
   #que les images soient applaties et donc les matrices résumées en 1 ligne
   # 2- On split le data set en 2 jeux de données : train et test
   # 3- On optimise les paramètres du RandomForrestClassifier (variables 
   #qualitatives) afin de trouver le modèle avec la meilleure accuracy
   # 4- On retient le modèle optimal
   # 5- On lance les prédictions sur le jeu de données test
   # 6- On affiche la matrice de confusion
   
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, metrics
from sklearn import model_selection
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
###############################################################################

###############################################################################       
###  PARAMETRAGE & CREATION DU RF
###############################################################################
# on recharge les données pour les adapter aux modèles RF et KNN
X, y = load_data(data)
#flattening du X
X = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))

# création train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

### Optimisation du nombre d'estimateurs du rf
#on définit le rf ainsi que les paramètres que l'on veut tester
rfc = RandomForestClassifier(random_state = 42)
param_grid_RF = { 
    'n_estimators': [200, 210],
    'max_depth' : [14, 16, 18]
}

#recherche des meilleurs paramètres
CV_rfc = model_selection.GridSearchCV(estimator = rfc, param_grid = param_grid_RF, cv = 5)
CV_rfc.fit(X_train, y_train)

#récupération des meilleurs paramètres
CV_rfc.best_params_

#meilleurs paramètres : {'max_depth': 14, 'n_estimators': 200}
# pour 0.75 d'accuracy

#Modèle optimal de Random Forest
# n_estimator =
# max_depth =

rf_opt = RandomForestClassifier(n_estimators = 200, max_depth = 14, criterion= "gini")
rf_opt.fit(X_train,y_train)

predictions_rf = rf_opt.predict(X_test)

#Affichage de la matrice de confusion (sur l'échantillon test)
conf_rf = confusion_matrix(y_test, predictions_rf)
pd.DataFrame(conf_rf, index = ["Pop_données", "Rock_données", "Rap_données", 
                            "Techno_données", "Classique_données"],
             columns = ["Pop_pred", "Rock_pred", "Rap_pred", "Techno_pred", 
                        "Classique_pred"])

                                                    

###############################################################################
### K NEAREST NEIGHBOURS
###############################################################################

###############################################################################
### SCRIPT KNN :
   
   # 1- On charge le jeu de données (déjà chargé dans la partie précédente RF)
   # 2- Séparation du jeu de données en test-train (partie précédente)
   # 3- On cherche les paramètres de la fonction KNeighborsClassifier qui
   # optimisent l'accuracy du modèle
   # 4- On crée le modèle avec les paramètres obtenus
   # 5- On lance les prédictions sur le jeu de données test et on affiche 
   # la matrice de confusion
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, metrics
from sklearn import model_selection
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
###############################################################################

###############################################################################       
###  PARAMETRAGE & CREATION DU KNN
###############################################################################       

# On fixe les valeurs des hyperparamètres à tester

param_grid = {'n_neighbors': [3, 5, 7, 10]}

# Choisir un score à optimiser, ici l'accuracy (proportion de prédictions correctes)
score = 'accuracy'

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
# avec la fonction KneighborsClassifier()
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # un classifieur kNN
    param_grid,     # hyperparamètres à tester
    cv=5,           # nombre de folds de validation croisée
    scoring = score   # score à optimiser
)

#On ce classifieur sur le jeu d'entraînement
clf.fit(X_train, y_train)

# On affiche le(s) hyperparamètre(s) optimaux
print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)

#Résultats : n_ neighbors optimaux = 3

## Création du modèle optimal

#instanciation et définition du k
model_knn = KNeighborsClassifier(n_neighbors = 3)
#fitting 
model_knn.fit(X_train,y_train)

#permet de donner le score de bonne prédiction
model_knn.score(X_test, y_test)

#correspondance des genres
list_corresp = dict(zip([0,1,2,3,4], ["Pop", "Rock", "Rap", "Techno", "Classique"]))
#Prediction 

prediction_knn = model_knn.predict(X_test)

#matrice de confusion

conf_knn = confusion_matrix(y_test, prediction_knn)
pd.DataFrame(conf_knn, index = ["Pop_données", "Rock_données", "Rap_données", 
                            "Techno_données", "Classique_données"],
             columns = ["Pop_pred", "Rock_pred", "Rap_pred", "Techno_pred", 
                        "Classique_pred"])

###############################################################################
