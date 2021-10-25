# -*- coding: utf-8 -*-
"""
Projet RakuPy

Application Streamlit
"""

import streamlit as st
from PIL import Image
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import tensorflow as tf
#import re
#import unicodedata

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from nltk.tokenize.regexp import RegexpTokenizer
#from nltk.corpus import stopwords
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras import backend as K
from nltk.tokenize import word_tokenize
from joblib import dump, load


##### Chargement des données #####

@st.cache
def load_data():
    features = pd.read_csv("Data/textes/X_train.csv",index_col=0)
    return features
df = load_data()

@st.cache
def load_target():
    y_tr = pd.read_csv("Data/y_train.csv",index_col=0)
    return y_tr
target = load_target()

data = pd.concat([df,target],axis=1)


##### fonction pour rogner/redimensionner #####
def crop_resize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts)==0:
        img_crop = img
    else:
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x,y,w,h = cv2.boundingRect(cnt)
        img_crop = img[y:y+h, x:x+w]
    img_resize = cv2.resize(img_crop, dsize=(350,350), interpolation=cv2.INTER_CUBIC)
    return img_resize


##### Jeu d'entraînement et jeu de validation #####
data_train, data_valid, y_train, y_valid = train_test_split(df,target,test_size=13812,stratify=target,random_state=34)

def creation_chemin(col1,col2):
     return '/Volumes/GoogleDrive/Mon Drive/Data/images/image_train/image_' + str(col1) + '_product_' + str(col2) + '.jpg'
data_valid['img_paths'] = np.vectorize(creation_chemin)(data_valid['imageid'],data_valid['productid'])
data_train['img_paths'] = np.vectorize(creation_chemin)(data_train['imageid'],data_train['productid'])


##### Correspondance entre les indices et les classes ####
list_classe = ['10', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', '2705', '2905', '40', '50', '60']
name_classe = ["Livres d'occasion","Produits dérivés","Cartes à jouer, à collectionner","Déguisement, figurines, accessoires de jeux",
               "Jouets","Jouets","Drones, modèles réduits","Vêtement et accessoires enfants","Jeux enfants et adolescents",
               "Petite enfance","Mobilier intérieur","Accessoires intérieurs","Produits alimentaires","Décoration d'intérieur",
               "Produits pour animaux","Journaux, magazines","Livres","Consoles de jeux, jeux vidéos","Papèterie",
               "Mobilier extérieur, jardin","Piscines","Jardinage, bricolage","Livres, romans","Jeux vidéos en téléchargement",
               "Jeux vidéos d'occasion","Accessoires jeux vidéos d'occasion","Consoles de jeux vidéos d'occasion"]
correspondance = pd.concat([pd.Series(list_classe),pd.Series(name_classe)],axis=1)

##### Barre latérale #####
st.sidebar.title('RakuPy')
st.sidebar.subheader('Menu')
liste_choix = ['Le projet RakuPy',
               'Jeu de données',
               'Analyse du jeu de données',
               'Notre modèle',
               'Résultats',
               'Prédictions sur le jeu de validation',
               'Démo']
choix = st.sidebar.radio('',liste_choix)

st.sidebar.subheader('RakuTeam')

st.sidebar.info('''
                Gilles Nicolet        
                https://www.linkedin.com/in/gilles-nicolet/
                
                Abdelhadi Serbouti            
                https://www.linkedin.com/in/abdelhadi-serbouti-a083779a/
                
                Benjamin Vignau             
                https://fr.linkedin.com/in/benjamin-vignau-0479b916   
                  
                https://datascientest.com     
                Formation DS Bootcamp            
                Promotion Août 2021''')



##### Présentation du projet #####
if choix==liste_choix[0]:
    st.title('Le projet RakuPy')
    st.subheader("Classification automatique d'articles e-commerce")
    
    st.markdown('La catégorisation des produits est un problème crucial pour les sites d’e-commerce. En effet, il est indispensable de pouvoir proposer aux clients des produits correspondants à leurs recherches ainsi que leur faire des recommandations personnalisées aussi pertinentes que possible. Ensuite, plus un produit est mal étiqueté et plus il aura du mal se vendre.')
    st.markdown('Une des difficultés de cette tâche est que les grandes plateformes d’e-commerce regroupent des vendeurs variés (y compris des non professionnels) pouvant décrire des produits similaires de manières très différentes, voire dans des langues différentes. De plus, il existe souvent un grand nombre de catégories possibles pour ces produits.')
    st.markdown('Ce projet a ainsi pour objectif de proposer un modèle permettant de classer automatiquement les produits provenant d’une plateforme de e-commerce à partir de deux sources : leur image et le texte présentant ces produits.')
    st.markdown("Un tel modèle pourrait servir à proposer une ou plusieurs catégories aux vendeurs pour limiter les erreurs lorsqu'ils remplissent leurs annonces sur la plateforme. Il serait également possible d'identifier des produits potentiellement mal étiquetés afin de conseiller les vendeurs de vérifier leurs annonces.")

    @st.cache
    def load_figure_1():
        img = Image.open('Figures/Figure_Rakuten.png') 
        return img
    
    st.image(load_figure_1(),caption="Capture d'écran du site de Rakuten France : nous pouvons voir que la classification des produits en catégories et en sous-catégories est omniprésente.")



##### Description du jeu de données #####
if choix==liste_choix[1]:
    st.title('Jeu de données')
    
    st.markdown("Pour entraîner et évaluer notre modèle, nous avons utilisé des données concernant des produits du catalogue de Rakuten France. Ces jeux de données sont issus du concours Rakuten Data Challenge 2020.  Ils sont disponibles sur le site du Challenge :")
    st.markdown('https://challengedata.ens.fr/challenges/35')
    st.markdown("Nous pouvons y trouver notamment un jeu de données sur 84916 produits. Pour chacun des ces produits, nous avons des données textuelles et une image d'illustration. Nous avons séparé ces données en un jeu d'entraînement et un jeu de validation de tailles respectives 71104 et 13812. Un aperçu est disponible ci-dessous.")
    st.markdown('Les colonnes de ce tableau sont :')
    st.markdown(''' 
                * _designation_ : les titres des produits
                * _description_ : les descriptions des produits (pas utilisées dans notre modèle)
                * Les colonnes _productid_ et _imageid_ permettent d’accéder aux noms des fichiers images
                * _prdtypecode_ : codes des catégories des produits (variable cible)
                ''')
    
    st.dataframe(data[:10000])


    
    
    
##### Dataviz ######
if choix==liste_choix[2]:
    st.title('Analyse du jeu de données')
 
    st.markdown("Les produits sont classés en 27 catégories. Comme nous pouvons le voir dans la figure ci-dessous, ces catégories sont très déséquilibrées. La plus grande comprend 10209 produits. Les autres ont des effectifs compris entre 764 et 5073 articles.")
    
    @st.cache
    def load_figure_2():
        img = Image.open('Figures/countplot_product_type_code.jpeg') 
        return img
    
    st.image(load_figure_2(),caption="Figure : nombre de produits par catégorie")

    st.markdown("Comme nous pouvons le remarquer en parcourant les données, beaucoup de vendeurs ne remplissent pas la partie description. Environ 35% des produits n'ont pas de description.")
    
    st.markdown('''Les catégories des produits (variable cible) ne sont disponibles qu'à travers un code chiffré sans plus de précisions. 
                Cependant, nous pouvons voir ci-dessous que les nuages des mots les plus fréquents permettent d'avoir une idée plutôt précise du contenu de la plupart des catégories. 
                Il reste toutefois quelques catégories pour lesquelles le nuage de mots ne permet pas une identification sûre.
                En complètant cette analyse des mots les plus fréquents par un visualisation de quelques images pour chacune des catégories, nous avons pu mettre des mots sur toutes les catégories.
                ''')
    
    options_selectbox = ['Catégories 1-9','Catégories 10-18','Catégories 19-27']
    choix_select = st.selectbox('Catégories pour les nuages de mots',options_selectbox)
    if choix_select==options_selectbox[0]:
        @st.cache
        def load_figure_3_1():
            img = Image.open('Figures/wordcloud_categories_1.jpeg') 
            return img
    
        st.image(load_figure_3_1(),caption="Figure : nuages de mots pour les 9 premières catégories")
        
    if choix_select==options_selectbox[1]:
        @st.cache
        def load_figure_3_2():
            img = Image.open('Figures/wordcloud_categories_2.jpeg') 
            return img
    
        st.image(load_figure_3_2(),caption="Figure : nuages de mots pour les 9 catégories du milieu")
        
    if choix_select==options_selectbox[2]:
        @st.cache
        def load_figure_3_3():
            img = Image.open('Figures/wordcloud_categories_3.jpeg') 
            return img
    
        st.image(load_figure_3_3(),caption="Figure : nuages de mots pour les 9 dernières catégories")

    st.markdown('Les images accompagnant les produits de la plateforme sont très variées. La qualité des images est très inégale. Certaines peuvent présenter des bordures blanches.')
    
    indice_produit_0 = st.slider('Choisissez un produit',1,71104)

    im_illustr = plt.imread(data_train.iloc[indice_produit_0-1,4])
    text_illustr = data_train.iloc[indice_produit_0-1,0]
    cat_illustr = y_train.iloc[indice_produit_0-1,0]
    classe_illustr = correspondance.iloc[:,1][correspondance.iloc[:,0]==str(cat_illustr)].iloc[0]

    st.text("Catégorie " + str(cat_illustr) + " (" + classe_illustr + ")")
    st.text(text_illustr)
    st.image(im_illustr,caption="Figure : Articles du jeu de données avec titre, image et catégorie")


 
   
##### Présentation du modèle ######
if choix==liste_choix[3]:
    st.title('Notre modèle')
    
    st.markdown('''Notre modèle est composé de deux parties. Premièrement avec composante texte formée par un réseau de neurones convolutionel (CNN).
                Ce CNN comprend notamment 6 canaux convolution-max pooling placés en parallèle afin de détecter des combinaisons de mots de différentes longueurs.''')

    @st.cache
    def load_figure_5():
        img = Image.open('Figures/CNN.png') 
        return img
        
    st.image(load_figure_5(),caption="Figure : Description du modèle CNN pour le texte")
    
    st.markdown('''La composante image de notre modèle est formée par la partie convolutionnelle du modèle EfficientNetB4 préentraînée sur le célèbre jeu de données Imagenet.
                à laquelle nous avons rajouté une couche dense de 512 neurones.''')
    
    @st.cache
    def load_figure_6():
        img = Image.open('Figures/EfficientNetB4.png') 
        return img
        
    st.image(load_figure_6(),width=200,caption="Figure : Description du modèle EfficientNetB4 pour les images")
 
    st.markdown('''Pour former notre modèle mixte textes-images nous concaténons ces deux composantes juste avant la couche dense de sortie.
                La différence de taux entre les couches de Dropout de sortie des deux parties du modèle (0.4 pour les textes et 0.2 pour les images)
                nous permet de mieux prendre en compte la composante images par rapport à la composante textes.''')
                
    @st.cache
    def load_figure_7():
        img = Image.open('Figures/Modele_mixte.png') 
        return img
        
    st.image(load_figure_7(),width=300,caption="Figure : Description du modèle mixte")
    
    
    
##### Résultats du modèle ######
if choix==liste_choix[4]:
    st.title('Résultats')
    
    st.markdown('Nous présentons ici les résultats de notre modèle sur le jeu de validation.')
    
    st.markdown('''La première figure présente les scores de notre modèle. Nous présentons la précision, le rappel et et le F1-score) pour chacune des catégories de produits.
                Nous affichons également les scores globaux. Notre modèle affiche notament une accuracy de 0.83 et un F1-score de 0.83.
                Le meilleure score (F1-score=0.99) est obtenu pour la catégorie des jeux en téléchargement.
                Les catégories posant le plus de problèmes à notre modèle sont les catégories 1281 (F1-score=0.56) et 10 (0.61)
                pour des raisons expliquées ci-dessous avec la matrice de confusion.''')
    
    @st.cache
    def load_figure_8():
        img = Image.open('Figures/Resultats.png') 
        return img
    st.image(load_figure_8(),width=500,caption="Figure : Scores du modèle")

    st.markdown('''La seconde figure montre la matrice de confusion. Nous pouvons remarquer que la principale difficulté pour notre modèle
                est de distinguer les catégories voisines comme les deux catégories de jouets 1280 et 1281 ou les deux catégories de livres 10 et 2705.''')

    @st.cache
    def load_figure_9():
        img = Image.open('Figures/Confusion_matrix.jpg') 
        return img
    st.image(load_figure_9(),width=850)

    
    
###### Demo sur le jeu de test #####    
if choix==liste_choix[5]:
    st.title('Prédictions sur le jeu de validation')
    
    indice_produit = st.slider('Choisissez un produit dans le jeu de validation',1,13812)

    ##### Chargement du Tokenizer #####
    @st.cache(show_spinner=False)
    def tokenize():
        #tokenizer = load('/Volumes/GoogleDrive/Mon Drive/Models/tokenizer.joblib') 
        tokenizer = load('Models/tokenizer.joblib') 
        return tokenizer

    tokenizer = tokenize()

    ##### Preprocessing pour le jeu de test #####
    @st.cache()
    def preprocessing_text():
        X_valid = tokenizer.texts_to_sequences(data_valid.designation)
        X_valid = pad_sequences(X_valid,padding='post',truncating='post',maxlen=34)
        return X_valid

    X_valid = preprocessing_text()

    ##### Chargement du modele #####
    #@st.cache(hash_funcs={'keras.utils.object_identity.ObjectIdentityDictionary': lambda _: None})
    def load_nn():
        url = 'https://drive.google.com/file/d/10-i9BY56IiO-4lApwLv4vMJnqioouEvj/view?usp=sharing'
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        model = load_model(path)
        #model = load_model('/Volumes/GoogleDrive/Mon Drive/Models/EfficientNetB4_CNN_2.h5')
        return model

    model = load_nn()
    
    text = X_valid[indice_produit-1]
    im = plt.imread(data_valid.iloc[indice_produit-1,4])

    img = crop_resize(im)
    img = img.reshape((1,350,350,3))
    text = text.reshape((1,34,))
    titre = data_valid.iloc[indice_produit-1,0]
    
    prediction = model.predict(x=[img,text])
    y_true = y_valid.iloc[indice_produit-1,0]
    classe_true = correspondance.iloc[:,1][correspondance.iloc[:,0]==str(y_true)].iloc[0]
    y_pred = list_classe[prediction.argmax()]
    classe_pred = name_classe[prediction.argmax()]
    proba = prediction.max().round(2)

    st.text(titre)
    st.image(im)

    st.markdown('Prédiction :')
    st.text(y_pred + ' (' + classe_pred + ')')
    st.markdown('Probabilité :')
    st.text(proba)
    st.markdown('Vraie Catégorie :')
    st.text(str(y_true) + ' (' + classe_true + ')')
    
        

###### Demo #####    
if choix==liste_choix[6]:
    st.title('Démo')

    titre_0 = st.text_input("Titre")
    image_0 = st.file_uploader("Image")

    clicked_2 = st.button('Prédire')
    
    if clicked_2:
        ##### Chargement du modele #####
        #@st.cache(hash_funcs={'keras.utils.object_identity.ObjectIdentityDictionary': lambda _: None})
        def load_nn():
            url = 'https://drive.google.com/file/d/10-i9BY56IiO-4lApwLv4vMJnqioouEvj/view?usp=sharing'
            path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
            model = load_model(path)
            #model = load_model('/Volumes/GoogleDrive/Mon Drive/Models/EfficientNetB4_CNN_2.h5')
            return model
    
        model = load_nn()
    
        ##### Chargement du Tokenizer #####
        @st.cache(show_spinner=False)
        def tokenize():
            #tokenizer = load('/Volumes/GoogleDrive/Mon Drive/Models/tokenizer.joblib') 
            tokenizer = load('Models/tokenizer.joblib') 
            return tokenizer

        tokenizer = tokenize()

        ##### Preprocessing  #####
        @st.cache()
        def preprocessing_text_2():
            titre_1 = tokenizer.texts_to_sequences([titre_0,titre_0])
            titre_2 = pad_sequences(titre_1,padding='post',truncating='post',maxlen=34)
            return titre_2[0,]

        texte_1 = preprocessing_text_2()
        texte_1 = texte_1.reshape((1,34,))

        image_1 = crop_resize(plt.imread(image_0))
        image_1 = image_1.reshape((1,350,350,3))
    
        prediction = model.predict(x=[image_1,texte_1])
        y_pred = list_classe[prediction.argmax()]
        classe_pred = name_classe[prediction.argmax()]
        proba = prediction.max().round(2)

        st.image(image_0)

        st.markdown('Prédiction :')
        st.text(y_pred + ' (' + classe_pred + ')')
        st.markdown('Probabilité :')
        st.text(proba)
    

