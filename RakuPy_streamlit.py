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

st.sidebar.info('RakuTeam\n Gilles Nicolet\n Abdelhadi Serbouti \n Benjamin Vignau')

#st.sidebar.text('Gilles Nicolet')

#st.sidebar.text('Abdelhadi Serbouti')

#st.sidebar.text('Benjamin Vignau')



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

image = 

#st.cache()
#def train_model():
    
#    return
    





