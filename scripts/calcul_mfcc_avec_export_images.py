import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

# fonction pour calculer la moyenne de la dérivée première d'un ensemble de valeurs ("delta")
def mean_delta(values, axis=0):
  delta = np.diff(values, axis=axis)
  return(np.mean(delta, axis=axis))

# fonction pour calculer la moyenne de la dérivée seconde d'un ensemble de valeurs ("deltadelta")
def mean_deltadelta(values, axis=0):
  delta = np.diff(np.diff(values, axis=axis), axis=axis)
  return(np.mean(delta, axis=axis))

# fonction pour exporter un tableau 2D numpy sous forme d'image en niveaux de gris
def export_grayscale_image(numpy_data, filename, image_width_inches, image_height_inches, resolution_dpi=None, vmin=None, vmax=None):
  fig = plt.figure(frameon=False)
  fig.set_size_inches(image_width_inches,image_height_inches)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(numpy_data, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
  fig.savefig(filename, dpi = resolution_dpi)
  plt.close(fig)

##############################

## definition des parametres
# dossiers contenant les extraits audio à analyser
dossiers_extraits_audio = ['output_audio_test']
# calcul des MFCC
duree_max_signal_secondes = 2.5 # None pour utiliser la durée de l'extrait le plus long présent dans les données
n_coefs_MFCC = 13
duree_trame_millisecondes = 25
recouvrement_entre_trames_millisecondes = 10
# export images
dossier_images = "imagesMFCC"
sousdossier_images_valeurs_brutes = "brut"
sousdossier_images_valeurs_normalisees = "norm"
sousdossier_images_valeurs_brutes_aleatoire = "brut_rnd"
sousdossier_images_valeurs_normalisees_aleatoire = "norm_rnd"
extension_images = ".png"
largeur_images_pouces = 3
hauteur_images_pouces = .5
resolution_images_dpi = 50
zero_padding_images = True # si False, les images plus petites (extraits audios plus courts) sont étirées pour atteindre la même largeur
zero_padding_blanc = True # si False, on complète avec des pixels noirs

##############################

## calcul des MFCC sur chaque extrait audio et stockage des valeurs dans un tableau

dfRawValues = pd.DataFrame()
for targetdir in dossiers_extraits_audio:
  for path in os.listdir(targetdir):
      print("%s - %s" % (targetdir, path))
      X, sample_rate = librosa.load(targetdir + '/' + path
                                    ,res_type='kaiser_fast'
                                    ,duration=duree_max_signal_secondes
                                    ,sr=None # frequence d'echantillonnage determinee par librosa.load a partir de l'en-tete du fichier .wav
                                    ,offset=0.0
                                   )
      sample_rate = np.array(sample_rate)

      # calcul des 13 coefs MFCC sur chaque trame
      mfccs_raw = np.transpose(librosa.feature.mfcc(y=X,
        sr=sample_rate, 
        n_mfcc=n_coefs_MFCC,
        n_fft=int(sample_rate*duree_trame_millisecondes/1000), # extraction des MFCC sur des trames de duree_trame_millisecondes ms (cf. https://github.com/librosa/librosa/issues/584)
        hop_length=int(sample_rate*(duree_trame_millisecondes-recouvrement_entre_trames_millisecondes)/1000))) # definition du decalage entre trames consecutives (cf. https://github.com/librosa/librosa/issues/584)
      
      # stockage des informations dans le tableau dfRawValues avec les infos sur le nom de l'extrait audio
      mfccs_raw_df = pd.DataFrame(mfccs_raw)
      mfccs_raw_df['extrait'] = path
      dfRawValues = pd.concat([dfRawValues, mfccs_raw_df])

##############################

## normalisation des valeurs pour chacun des coefficients MFCC

# calcul de la moyenne et de l'ecart-type de chacun des coefficients MFCC
npRawValues = dfRawValues.drop(columns = ['extrait']).to_numpy()
npRawValuesColMeans = np.mean(npRawValues, axis = 0)
npRawValuesColStds = np.std(npRawValues, axis = 0)

# normalisation en z-scores de chacune des colonnes (soustraction de la moyenne puis division par l'ecart-type)
npNormValues = (npRawValues - npRawValuesColMeans) / npRawValuesColStds

# on remet les valeurs normalisees dans une DataFrame avec la colonne extrait
dfNormValues = pd.DataFrame(npNormValues)
dfNormValues['extrait'] = dfRawValues['extrait'].tolist()

##############################

## export dans des fichiers textes des valeurs brutes et normalisées des MFCC pour chaque trame

dfRawValues.to_csv('valeurs_MFCC_brutes.txt', sep = '\t', index = False)
dfNormValues.to_csv('valeurs_MFCC_norm.txt', sep = '\t', index = False)

##############################

## calcul des statistiques pour chaque extrait audio sur les valeurs brutes et normalisées

## valeurs brutes
# calcul des statistiques sur chaque extrait pour chacun des coefficients MFCC
statsRaw = dfRawValues.groupby(['extrait']).agg(['mean', 'std', mean_delta, mean_deltadelta])
# conversion en colonnes du critère de regroupement extrait (par défaut indices de lignes)
statsRaw.reset_index(level=statsRaw.index.names, inplace=True)
# fusion des 2 niveaux hiérarchiques des noms de colonnes (indice des MFCC et fonction d'aggrégation) en un seul niveau
# formatage conditionnel pour tenir compte des colonnes avec un seul élément
statsRaw.columns = [f"{x}" if not y else f"{x}_{y}" for x, y in statsRaw.columns.to_flat_index()]
# export du vecteur de paramètres pour chaque extrait
statsRaw.to_csv('stats_MFCC_brutes.txt', sep = '\t', index = False)

## valeurs normalisées
# calcul des statistiques sur chaque extrait pour chacun des coefficients MFCC
statsNorm = dfNormValues.groupby(['extrait']).agg(['mean', 'std', mean_delta, mean_deltadelta])
# conversion en colonnes du critère de regroupement extrait (par défaut indices de lignes)
statsNorm.reset_index(level=statsNorm.index.names, inplace=True)
# fusion des 2 niveaux hiérarchiques des noms de colonnes (indice des MFCC et fonction d'aggrégation) en un seul niveau
# formatage conditionnel pour tenir compte des colonnes avec un seul élément
statsNorm.columns = [f"{x}" if not y else f"{x}_{y}" for x, y in statsNorm.columns.to_flat_index()]
# export du vecteur de paramètres pour chaque extrait
statsNorm.to_csv('stats_MFCC_norm.txt', sep = '\t', index = False)

##############################

## export des images pour chacun des fichiers audio

# liste des extraits
listeExtraits = dfRawValues[['extrait']].drop_duplicates()

# valeurs minimum et maximum dans l'ensemble des données (version brute et normalisée) à prendre en compte lors du tracé des images
vmin_brut = np.amin(npRawValues)
vmax_brut = np.amax(npRawValues)
vmin_norm = np.amin(npNormValues)
vmax_norm = np.amax(npNormValues)
# valeur correspondant au blanc (ou noir) pour le zero-padding : minimum ou maximum dans l'ensemble des données
if zero_padding_blanc: # blanc = valeur max
  valeur_zero_padding_brut = vmax_brut
  valeur_zero_padding_norm = vmax_norm
else: # noir = valeur min
  valeur_zero_padding_brut = vmin_brut
  valeur_zero_padding_norm = vmin_norm

if zero_padding_images:
  # taille maximale en nombre de trames pour le "zero-padding" des images plus petites
  nMaxTrames = dfRawValues.groupby(['extrait'])[0].count().max()

# création des dossiers de destination si nécessaire
Path(dossier_images).mkdir(parents=True, exist_ok=True)
# un sous-dossier par type d'image (données brutes ou normalisées, avec ou sans ordre aléatoire)
target_subfolders = [sousdossier_images_valeurs_brutes, sousdossier_images_valeurs_normalisees, sousdossier_images_valeurs_brutes_aleatoire, sousdossier_images_valeurs_normalisees_aleatoire]
for subfolder in target_subfolders:
  Path(dossier_images+'/'+subfolder).mkdir(parents=True, exist_ok=True)
for index, row in listeExtraits.iterrows():
  extrait = row['extrait']
  # traitement des valeurs brutes
  npRawExtraitCourant = dfRawValues[dfRawValues['extrait']==extrait].drop(columns = ['extrait']).to_numpy()
  # version avec les trames mélangées en ordre aléatoire
  npRawExtraitCourantRnd = np.copy(npRawExtraitCourant)
  np.random.shuffle(npRawExtraitCourantRnd)

  # ajout de zéros a la fin si le nombre de lignes est inférieur au maximum ("zero padding")
  if zero_padding_images and npRawExtraitCourant.shape[0]<nMaxTrames:
    npRawExtraitCourant = np.append(npRawExtraitCourant, np.ones((nMaxTrames-npRawExtraitCourant.shape[0], 13))*valeur_zero_padding_brut, axis=0)
    npRawExtraitCourantRnd = np.append(npRawExtraitCourantRnd, np.ones((nMaxTrames-npRawExtraitCourant.shape[0], 13))*valeur_zero_padding_brut, axis=0)

  # traitement des valeurs normalisées
  npNormExtraitCourant = dfNormValues[dfNormValues['extrait']==extrait].drop(columns = ['extrait']).to_numpy()
  # version avec les trames mélangées en ordre aléatoire
  npNormExtraitCourantRnd = np.copy(npNormExtraitCourant)
  np.random.shuffle(npNormExtraitCourantRnd)

  # ajout de zéros a la fin si le nombre de lignes est inférieur au maximum ("zero padding")
  if zero_padding_images and npNormExtraitCourant.shape[0]<nMaxTrames:
    npNormExtraitCourant = np.append(npNormExtraitCourant, np.ones((nMaxTrames-npNormExtraitCourant.shape[0], 13))*valeur_zero_padding_norm, axis=0)
    npNormExtraitCourantRnd = np.append(npNormExtraitCourantRnd, np.ones((nMaxTrames-npNormExtraitCourant.shape[0], 13))*valeur_zero_padding_norm, axis=0)

  # construction et export des images dans les sous-dossiers cibles
  export_grayscale_image(npRawExtraitCourant.transpose(), dossier_images+"/"+sousdossier_images_valeurs_brutes+"/"+extrait+extension_images, largeur_images_pouces, hauteur_images_pouces, resolution_images_dpi, vmin_brut, vmax_brut)
  export_grayscale_image(npRawExtraitCourantRnd.transpose(), dossier_images+"/"+sousdossier_images_valeurs_brutes_aleatoire+"/"+extrait+extension_images, largeur_images_pouces, hauteur_images_pouces, resolution_images_dpi, vmin_brut, vmax_brut)
  export_grayscale_image(npNormExtraitCourant.transpose(), dossier_images+"/"+sousdossier_images_valeurs_normalisees+"/"+extrait+extension_images, largeur_images_pouces, hauteur_images_pouces, resolution_images_dpi, vmin_norm, vmax_norm)
  export_grayscale_image(npNormExtraitCourantRnd.transpose(), dossier_images+"/"+sousdossier_images_valeurs_normalisees_aleatoire+"/"+extrait+extension_images, largeur_images_pouces, hauteur_images_pouces, resolution_images_dpi, vmin_norm, vmax_norm)
