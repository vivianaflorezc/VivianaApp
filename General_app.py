import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from prueba1 import Algeria_app
#from data_afric import Animal_app
import streamlit as st
#from streamlit_option_menu import option_menu

with st.sidebar:
    modelo = st.selectbox('Select an option',('Algeria','Animal'))

# Mostrar una vista diferente para cada modelo seleccionado
if modelo == 'Algeria':
    Algeria_app()

elif modelo == 'Animal':
    Animal_app()

