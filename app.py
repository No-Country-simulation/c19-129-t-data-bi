import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from scipy.stats import boxcox
import zipfile as zp

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.subplots as sp

# Page configuration
st.set_page_config(page_title="EligentIA🕵🏻", layout="wide", page_icon="🔎")

# Title
st.title("EligentIA 🕵🏻")

# Load the model
model = joblib.load("model/best_model.joblib")# Random Sampling

 

st.header('Model')
st.markdown('Introduce the values for the transaction you would like to evaluate', unsafe_allow_html=True)

@st.cache_resource
def prediction(Account_length, Age, Total_income, Years_employed) :
     # New dataset with the input data
    input_data = pd.DataFrame([[Account_length, Age, Total_income, Years_employed]], 
                                    columns=['Account_length', 'Age', 'Total_income', 'Years_employed'])
    
     # Make the prediction
    prediction = model.predict(input_data)

    if prediction == 0:
        pred = 'Aprobado'
    else:
        pred = 'Desaprobado'
    return pred
    
        # Create the input features
            
Account_length= st.slider('Antigüedad de la cuenta', 0, 60, disabled=False)
Total_income = st.slider('Ingresos Anuales', 12000, 100000, disabled=False)
Age = st.slider('Edad', 18, 80, disabled=False)
Years_employed = st.slider('Años de empleo **Marque 0 si esta desempleado**', 0, 20, disabled=False)
result =""

# Button to make the prediction
if st.button("Predict"): 
    result = prediction(Account_length, Age, Total_income, Years_employed) 
    if result == 'Aprobado':
        st.success('Su calificacion para tarjeta de credito esta: {}'.format(result))
    else:
        st.error('Su calificacion para tarjeta de credito esta:  {}'.format(result))