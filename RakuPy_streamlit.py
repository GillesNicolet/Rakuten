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
choix = st.sidebar.multiselect(liste_choix)





#st.write('')

#st.dataframe('')

#st.cache()
#def train_model():
    
#    return


