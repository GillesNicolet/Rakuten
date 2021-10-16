# -*- coding: utf-8 -*-
"""
Projet RakuPy

Application Streamlit
"""

import streamlit as st

##### Barre latérale #####
st.sidebar.title('RakuPy')
st.sidebar.subheader('Menu')
liste_choix = ['Le projet',
               'Dataset',
               'Le modèle',
               'Demo']
choix = st.sidebar.radio('',liste_choix)

st.sidebar.subheader('RakuTeam')

st.sidebar.info('Gilles Nicolet')

st.sidebar.info('Abdelhadi Serbouti')

st.sidebar.info('Benjamin Vignau')



##### Présentation du projet #####
if choix==liste_choix[0]:
    st.title('Le projet RakuPy')
    st.subheader("Classification automatique d'articles e-commerce")

#st.write('')




##### Description du jeu de données #####
if choix==liste_choix[1]:
    st.title('Dataset')

#st.dataframe('')

   
    
    
##### Description du modèle ######
if choix==liste_choix[2]:
    st.title('Le modèle')
 
    
    
    
    
###### Demo #####    
if choix==liste_choix[3]:
    st.title('Demo')

    text = st.text_input("Intitulé de l'article")

    image = st.file_uploader("Image de l'article")

#st.cache()
#def train_model():
    
#    return
    





