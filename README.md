# Détection des accents en anglais à partir des données audio (USA/India/England)

## Résumé
Ce projet vise à détecter les accents anglais à partir des coefficients cepstraux en fréquence de Mel (MFCC) extraits d’extraits audio. Ces caractéristiques sont transformées en images de spectrogrammes, utilisées comme entrée pour des modèles d’apprentissage automatique et d’apprentissage profond. L’objectif principal est de classer les enregistrements audio selon trois accents différents : britannique, américain et indien/sud-asiatique. Trois modèles ont été utilisés : Random Forest, perceptron multicouche (MLP) et réseau de neurones convolutif (CNN).

Keming et Jinyu ont assuré la collecte, le prétraitement et l’analyse des données ainsi que la transformation en caractéristiques MFCC. Xinlei, Siman et Lidan se sont chargés de l’entraînement des modèles de détection à l’aide de trois algorithmes différents: Random Forest, MLP et CNN. Enfin, tous les membres du groupe ont participé à la rédaction de l’article récapitulatif en fonction de leurs responsabilités respectives.

## Méthodologie
1. Collecte et prétraitement des données
Dataset : Le dataset provient de Common Voice et contient des enregistrements audio de locuteurs britanniques, américains et indiens.
Étapes de prétraitement :
- Extraction des caractéristiques MFCC à l’aide de la bibliothèque Librosa.
- Normalisation des données avec un z-score.
- Conversion des données MFCC en images de spectrogrammes en niveaux de gris.
2. Entraînement des modèles
3. Évaluation
Le modèle CNN a obtenu la meilleure précision avec 65,3 %, surpassant Random Forest (60,9 %) et MLP (60,0 %).
Des matrices de confusion ont été générées pour analyser les erreurs de classification.
