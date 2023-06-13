# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:50:26 2022
@author:  marion.corbera, niels.andre, floriane.cornue

L'objectif de ce script est de prédire le genre musical d'une chanson/musique 
en utilisant l'un des 3 modèles (soit RF, KNN, ou CNN) construits avec le 
script: 'Pipeline_SupervisedLearning_CNN_KNN_RF.py'.

Ce script premet de prétraiter une chanson/ musique et de prédire son genre
musical. Le code est composé de deux parties:
    
-   La première partie décrit la "pipeline" de prétraitements appliquée à la 
    musique choisie.
-   La deuxième partie permet de prédire sont genre musical avec l'un des 3 
    modèles au choix (RF, KNN, ou CNN).

"""
###############################################################################
### PROCESSING PIPELINE 
###############################################################################

###############################################################################
### SCRIPT PREPROCESS :
   
   # 1- Chargement de la musique.
   # 2- Découpages les audio en échnatillons de 4 secondes.
   # 3- Preprocess de l'échantillon (normalisation, découpage, padding & extraction des features demandées).
   # 4- Stockage nom des séquences audios et des features dans un dictionnaire.
   
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
        "name": [], # nom de la séquence
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
       



def sequenceur(record, sr, seq_time, overlap, wanted_feature, frame_size, hop_length, n_mfcc, dataset, file_name):  
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
        dataset["name"].append(sample_name) # nom de la séquence
        dataset[str(wanted_feature)].append(features) # featrue extrait de la séquence
        
    return dataset           



    
def Pipeline_preprocessing(filepath, sr=22050, mono=True, seq_time=0.5, overlap=0.1, wanted_feature='MFCCs_delta', frame_size=2048, hop_length=512, n_mfcc=13):
   
    # filepath = chemin de la chanson à tester
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
    
                
    file_name = filepath.split("\\")[-1] #  on grade que dernier mot du cgemin = titre      
    # on récupère le file name et on enlève l'extension du fichier
    file_name = file_name.replace(".WAV", "")
    file_name = file_name.replace(".wav", "")

    # on load les audio par genre musicaux (= par sous dossiers)
    record = loader(filepath, sr=sr, mono=mono) 
        
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
                         file_name)
                            
    
    return dataset


###############################################################################       



###############################################################################       
###  PARAMETRAGE & LANCEMENT DU PIPELINE DE PREPROCESSING
###############################################################################       

# FILEPATH = le chemin de la chanson à tester
# SR = sample rate => on met la frequence d'échantillonage de base des échantillons (pour ne pas perdre en résolution)
# MONO = True car on a que 1 seul cannal
# SEQ_TIME = durée des séuqneces => 4 secondes nous permettent 
# OVER_LAP = durée du chevauchement entre les séquences
# WANTED_FEATURE = le feature que l'ont souhaite extraire dans les séquences
# FRAME_SIZE = taille de la fenêtre (gde frame size : aumengte freq resolution / baisse temps resolution VS petite frame size : baisse freq resolution / aumengtes temps resolution)
# HOP_LENGTH = déplacement dans fenêtre (nb de pts on décalle vers droite pour changer de frame)
# N_MFCCS = nb de point mel utilisée echelle


FILEPATH = r'/Users/niels/Desktop/Cours/M2/Projet machine learning/Le Wanski - Tarte à la myrtille.wav'
SR = 22050              
MONO = True
SEQ_TIME = 4
OVER_LAP = 0.05
WANTED_FEATURE = 'MFCCs_delta' 
FRAME_SIZE = 2048 
HOP_LENGTH = 512 
N_MFCCS = 13 


if __name__ == "__main__":
    data = Pipeline_preprocessing(FILEPATH, sr=SR, wanted_feature=WANTED_FEATURE, n_mfcc=N_MFCCS)    
      

###############################################################################




###############################################################################
### PREDICTION DU GENRE MUSICAL 
###############################################################################


###############################################################################       
### IMPORT PACKAGES
###############################################################################

import collections
from tensorflow import keras

###############################################################################       
### IMPORT MODELS
###############################################################################

modelCNN = keras.models.load_model("/Users/niels/Desktop/Cours/M2/Projet machine learning/Spydata/CNNmodel")
#Charger le modèle RF et KNN à la main


#### Fonction qui récupère prédiction pour un échantillons inconnue
# changer input sizeqd on aura les bon


def predict_newdata(model, X):
    
    # fonction pour pédire la classe d'un échantillon
    # model : CNN entrainé & retenu
    # X : échantillon 
    
    X = X[np.newaxis, ...]

    # prédit le genre musical le plus prbable
    prediction = model.predict(X)

    # récupère index avec la valeur max => genre le plus probable d'appartenance
    predicted_index = np.argmax(prediction, axis=1)

    return(predicted_index[0])


'''
Permet de retourner le genre correspondant avec le nombre d'occurences plutôt que 0,1,2,3,4
'''

def corresp(dict_pred):
    dict_genre = dict(zip([0,1,2,3,4], ["Pop", "Rock", "Rap", "Techno", "Classique"]))
    for k in dict_genre.keys():
        dict_pred[dict_genre[k]] = dict_pred[k]
        del dict_pred[k]
    return (dict_pred)

'''
Prédiction du genre d'une musique choisie avec le knn
'''

def predictionmusiqueKNN (datamusique):
    X = np.array(datamusique[str(WANTED_FEATURE)]) 
    X = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))
    predictionssamples = knn.predict(X)
    return(corresp(collections.Counter(predictionssamples)))

'''
Prédiction du genre d'une musique choisie avec le CNN
'''

def predictionmusiqueCNN (datamusique):
    model = modelCNN #Modèle sur les données delta MFCCS 13
    X = np.array(datamusique[str(WANTED_FEATURE)])
    predictionssamples = []
    for k in range (len(X)):
        prediction = predict_newdata(model, X[k])
        predictionssamples.append(prediction)
    return(corresp(collections.Counter(predictionssamples)))

'''
Prédiction du genre d'une musique choisie avec le RF
'''

def predictionmusiqueRF (datamusique):
    X = np.array(datamusique[str(WANTED_FEATURE)])
    X = X.reshape(X.shape[0], (X.shape[1]*X.shape[2]))
    predictionssamples = rf_opt.predict(X)
    return(corresp(collections.Counter(predictionssamples)))

        
'''
Prédiction du genre d'une musique choisie avec choix du modèle par l'utilisateur
'''

def choixmodeleprediction(datamusique):
    choix = input("Quel modèle de prédiction voulez-vous utiliser sur votre musique ? RF (random forest)? KNN (k plus proches voisins) ? CNN (réseau de neurones) ? ")
    if choix == "KNN" or choix == "knn":
        return(predictionmusiqueKNN(datamusique))
    elif choix == "RF" or choix == "rf":
        return(predictionmusiqueRF(datamusique))
    elif choix =="CNN" or choix == "cnn":
        return(predictionmusiqueCNN(datamusique))
    else:
        print("Vous n'avez pas rentré ce qu'il fallait, veuillez réessayer.")










