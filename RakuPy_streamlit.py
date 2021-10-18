# -*- coding: utf-8 -*-
"""
Projet RakuPy

Application Streamlit
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as img

##### Barre latérale #####
st.sidebar.title('RakuPy')
st.sidebar.subheader('Menu')
liste_choix = ['Le projet RakuPy',
               'Jeu de données',
               'Data visualization',
               'Notre modèle',
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
    
    @st.cache
    def load_figure_1():
        image = img.imread('/Users/gilles/Documents/GitHub/Rakuten/Figures/Figure_Rakuten.png')
        return image
    
    #st.image(plt.imshow(load_figure_1()))
    
    st.write('La catégorisation des produits est un problème important et complexe pour les sites d’e-commerce. En effet, il est indispensable de pouvoir proposer aux clients des produits correspondants à leurs recherches ainsi que leur faire des recommandations personnalisées aussi pertinentes que possible.')
    st.write('Une des difficultés de cette tâche est que les grandes plateformes d’e-commerce regroupent des vendeurs variés (y compris des non professionnels) pouvant décrire des produits similaires de manières très différentes, voire dans des langues différentes. De plus, il existe un grand nombre de catégories possibles pour ces produits.')
    st.write('Ce projet a ainsi pour objectif de proposer un modèle permettant de classer automatiquement les produits provenant d’une plateforme de e-commerce à partir de deux sources : leur image et le texte présentant ces produits.')




##### Description du jeu de données #####
if choix==liste_choix[1]:
    st.title('Jeu de données')
    
    #@st.cache
    #def load_data():
    #    bikes_data_path = Path() / 'data/bike_sharing_demand_train.csv'
    #    data = pd.read_csv(bikes_data_path)
    #    return data

#st.write('Ce projet s’inscrit dans le cadre du concours Rakuten Data Challenge 2020 proposé sur le site challengedata.ens.fr de l’ENS et du Collège de France. Créée au Japon en 1997, Rakuten est une des plus grandes plateformes de commerce en ligne avec plus de 1,3 milliard d’utilisateurs. ')

#st.dataframe('')

#st.write(Nous avons utilisé les jeux de données du concours Rakuten Data Challenge 2020. Ces données sont disponibles librement sur le site du Challenge (après inscription au concours) [1]. Il faut rappeler que les données sont la propriété de Rakuten Institute of Technology.
#Les données sont séparées en un jeu d’entraînement et un jeu de test qui contiennent respectivement 84916 et 13812 produits (86% et 14%). La variable cible est la catégorie du produit qui est représentée par un code (product type code). Cette variable n’est pas disponible pour le jeu de test qui n’a d’intérêt que pour soumettre des prédictions pour le concours.
#Pour prédire cette catégorie, nous avions accès à deux types de données : des données textuelles et des images. Les données textuelles sont le titre (designation) et la description du produit (description). Chaque produit est illustré par une image.
#Les données sont sous forme d’un dossier d’images sous format jpeg et d’un tableau csv à quatre colonnes : 
#o	‘designation’ : tous les articles ont un champ “designation” renseigné
#o	‘description’ : 30% des articles n’ont pas de description
#o	‘productid’ : chaque article est identifié par un code “productid” unique
#o	‘imageid’ : chaque article est identifié par un code “ imageid ” unique
#Les colonnes ‘productid’ et ‘imageid’ permettent d’accéder aux fichiers images dont les noms sont de la forme ‘image_[imageid]_product_[productid].jpeg en remplaçant [imageid] et [productid] par leurs numéros indiqués dans le tableau csv. Le jeu de données n’a pas besoin d’être nettoyé, même s’il est à noter qu’environ 30% des articles n’ont pas de description. La principale modification nécessaire est la création d’une colonne indiquant le nom du fichier image avec son chemin d’accès à partir des colonnes ‘designation’ et ‘description’.
#Le jeu de données ne présente aucun doublon.
#La variable à expliquer est disponible sous la forme d’un fichier csv à une colonne (‘prdtypecode’). Nous pouvons observer que sa distribution est très déséquilibrée avec une très grande classe avec environ 10209 produits, et d’autres classes de tailles très variables entre 764 et 5073 produits (figure 1).
#)
    
    
    
##### Datasiz ######
if choix==liste_choix[2]:
    st.title('Data visualization')
 
    
    
    
    
    
##### Datasiz ######
if choix==liste_choix[3]:
    st.title('Notre modèle')
 
    
    
    
    
###### Demo #####    
if choix==liste_choix[4]:
    st.title('Démo')

    text = st.text_input("Intitulé de l'article")

    image = st.file_uploader("Image de l'article")

#st.cache()
#def train_model():
    
#    return
    





