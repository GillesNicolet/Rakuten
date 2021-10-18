# -*- coding: utf-8 -*-
"""
Projet RakuPy

Application Streamlit
"""

import streamlit as st

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

st.sidebar.info('''Datascientest \n 
                Promotion DS Bootcamp - Août \n 
                Gilles Nicolet 
                Abdelhadi Serbouti 
                Benjamin Vignau''')



##### Présentation du projet #####
if choix==liste_choix[0]:
    st.title('Le projet RakuPy')
    st.subheader("Classification automatique d'articles e-commerce")

#st.write('')




##### Description du jeu de données #####
if choix==liste_choix[1]:
    st.title('Jeu de données')

#st.dataframe('')

   
    
    
    
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
    





