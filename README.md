# Classification-Musique
Petit projet qui présente comment créer une classification automatique les différents genre de musique

Nous vous proposons de construire un modèle permettant de classifier le genre musical de musiques/chansons (nous nous sommes limités aux 5 genres Rock, Pop, Rap, Techno et Classique, mais nous pourrons tout à fait vous expliquer comment ajouter de nouveau genres si vous êtes interéssé).

Dans un premier temps, vous pouvez directement utilisé l'un de nos trois modèles de prédictions en utilisant le script "Pipeline_PredictionGenre_CNN_KNN_RF.py" avec une musique de votre choix (attention celle-ci doit être au format ".wav"). Ce script vous permettera de la "transformée" et de prédire son genre avec le modèle que vous choisirez. Vous pouvez également utilier nos données déjà pré-traitées "data13delta_mfccs.spydata". 
Voici comment vous devez procéder:

Pipeline_PredictionGenre_CNN_KNN_RF :

1 - Importer à la main dans spyder data13delta_mfccs.spydata
2 - Charger les packages et toutes les fonctions jusqu'au paramètrage & lancement du pipeline de preprocessing
3 - Changer le filepath pour la musique que vous souhaitez tester (tarte à la myrtille par défaut, mais faites-vous plaisir)
4 - Lancer le paramètrage  & lancement du pipeline de preprocessing
5 - Changer les chemins d'accès du dossier du modèle CNN dans prédiction du genre musical et importer à la main dans spyder les 2 autres modles
6 - Charger toutes les fonctions et imports dans prèdiction du genre musical
7 - Lancer la fonction choixdumodeleprediction(data)  (par défaut votre musique s'appelle data)
8 - Choisissez le modèle de prédiction que vous voulez !



Dans un deuxième temps, vous pouvez totalement reconstruire un nouveau modèle de prédiction en refaisant tourner notre script "Pipeline_SuppervisedLearning_CNN_KNN_RF.py". Vous pouvez même télécharger de nouvelles musiques (toujours au format .wav !) et ajouter un nouveau genre si vous le désirez. Ce script commencera par prétraiter vos musiques et les transformer par défaut en delta2 de 13 MFFCs (par défaut mais vous pouvez vous amuser avec d'autres features). Puis vous pourrez entraîner un réseau de neurones convolutionel, un random forest ou bien un K nearest neighbours à classer vos musiques. Ensuite à vous sauvegarder votre modèle et. de le réutiliser dans "Pipeline_PredictionGenre_CNN_KNN_RF" comme vu précédement.

Pipeline_SuppervisedLearning_CNN_KNN_RF.py

1 - Téléchargez vos musiques en format .wav (ou utiliser les notre envoyer en .zip une fois dé-zippées) et les ranger dans des sous dossier par genres.
2 - Installez/Chargez les pachages et fonctions.
3 - Mettre votre DATAPATH (chemin vers le dossier contenant les sous-dossiers des genres musicaux avec les musiques).
4 - Vous pouvez changer les paramètres physiques pour l'importation des sons et extractions de features (mais les valeurs par défauts sont les plus adaptées).
5 - Lancez la pipeline de preprocessing
6 - Installez/Chargez les packages et fonctions du modèle que vous souhaitez construire (attention pour le CNN, n'oubliez pas que le nombre de neurones de sortie du réseau soit égal au nombre de genre musicaux que vous essayez d'apprendre à reconnaître).
7 - Sauvegardez votre modèle et utilisez le dans : "Pipeline_PredictionGenre_CNN_KNN_RF.py"

