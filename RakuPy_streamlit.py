# -*- coding: utf-8 -*-
"""
Projet RakuPy

Application Streamlit
"""

import streamlit as st

# Barre latérale
st.sidebar.title('RakuPy')
st.sidebar.subheader('Menu')
liste_choix = ['Le projet',
               'Dataset',
               'Le modèle',
               'Demo']
choix = st.sidebar.radio('',liste_choix)


if choix==liste_choix[0]:
    st.title('Le projet RakuPy')
    st.subheader("Classification automatique d'articles e-commerce")

#st.write('')

if choix==liste_choix[1]:
    st.title('Dataset')

#st.dataframe('')

    
if choix==liste_choix[2]:
    st.title('Le modèle')
 
    
if choix==liste_choix[3]:
    st.title('Demo')

#st.cache()
#def train_model():
    
#    return
    





