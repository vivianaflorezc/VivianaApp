import streamlit as st
#import xgboost
import pandas as pd
import numpy as np
import joblib
def hepatitis_app():
    st.markdown("<h1 style='text-align: center;'>Hepatitis prediction</h1>", unsafe_allow_html=True)
    st.image('hepatitisimage.jpg')
    st.markdown('---')
    texto = ('The Hepatitis Assessment Module is a critical component of our application, dedicated to aiding in the early detection and evaluation of hepatitis. This module allows healthcare professionals to input essential data related to liver health, including liver function tests, patient history, and other crucial parameters. Leveraging advanced machine learning algorithms, the module employs XGBoost classification model to assess the health of the liver and predict potential risks.'
             'By providing accurate and timely insights, the Hepatitis Module empowers healthcare providers to make well-informed decisions and enhance the chances of early detection and effective management of hepatitis. The mission is to provide a valuable tool that supports medical professionals in the battle against hepatitis and ensures the best possible outcomes for patients.\n\n'
             'If you wish to assess liver health and receive valuable insights, please enter the relevant information below.'
        )
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model = joblib.load('hepatitis_model.pkl')
    Age = st.number_input('Age', min_value=0, max_value=100, value=10, step=1)
    Sex = st.selectbox('Sex', ('Female', 'Male'))
    ALB = st.number_input('Albumin Blood Test',min_value=0.0,max_value=200.0,value=40.0,step=0.1)
    ALP = st.number_input('Alkaline phosphatase',min_value=0.0,max_value=500.0,value=70.0,step=0.1)
    ALT = st.number_input('Alanine Transaminase',min_value=0.0,max_value=500.0,value=30.0,step=0.1)
    AST = st.number_input('Aspartate Transaminase',min_value=0.0,max_value=500.0,value=30.0,step=0.1)
    BIL = st.number_input('Bilirubin',min_value=0.0,max_value=300.0,value=10.0,step=0.1)
    CHE = st.number_input('Acetylcholinesterase',min_value=0.0,max_value=20.0,value=10.0,step=0.1)
    CHOL = st.number_input('Cholesterol',min_value=0.0,max_value=10.0,value=5.0,step=0.1)
    CREA= st.number_input('Creatinine',min_value=0.0,max_value=2000.0,value=80.0,step=0.1)
    GGT=  st.number_input('Gamma-Glutamyl Transferase',min_value=0.0,max_value=700.0,value=40.0,step=0.1)
    PROT = st.number_input('Proteins',min_value=30.0,max_value=100.0,value=70.0,step=0.1)
    if st.button('Predict'):
        data = {
            'Age': Age,
            'ALB': ALB,
            'ALP': ALP,
            'ALT': ALT,
            'AST': AST,
            'BIL': BIL,
            'CHE':CHE,
            'CHOL':CHOL,
            'CREA':CREA,
            'GGT':GGT,
            'PROT':PROT
        }
        data_cat = {
            'Sex': 'm' if Sex == 'Male' else 'f'
        }

        df = pd.DataFrame.from_dict([data])
        df_cat = pd.DataFrame.from_dict([data_cat])
        encoder = joblib.load('ohe_cirr.pkl')
        encoded_data = encoder.transform(df_cat)
        df_cat = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        # Unir las variables numéricas y categóricas codificadas
        data_full = pd.concat([df.reset_index(drop=True), df_cat], axis=1)
        #st.write(data_full)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(data_full)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        probabilities = model.predict_proba(input_data_reshaped)

        disease_labels = {
            0: 'Normal (Blood Donor)',
            1: 'Suspected Blood Donor',
            2: 'Hepatitis',
            3: 'Fibrosis',
            4: 'Cirrhosis'
        }

        for i, prob_vector in enumerate(probabilities):
            st.write(f'Probabilities for Disease Prediction {i + 1}:')
            max_prob_index = prob_vector.argmax()
            max_prob_disease = disease_labels.get(max_prob_index, "Unknown")

            for j, prob in enumerate(prob_vector):
                disease_label = disease_labels.get(j, "Unknown")
                st.write(f'{disease_label}: {prob:.4f}')

            st.write(f'Diagnosis for Prediction {i + 1}: {max_prob_disease}')



