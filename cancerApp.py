import streamlit as st
import pandas as pd
import numpy as np
import joblib
def cancer_app():
    # Justificar y alinear el texto
    st.markdown("<h1 style='text-align: center;'>Breast Cancer prediction</h1>", unsafe_allow_html=True)
    st.image('cancerimage.jpg')
    st.markdown('---')
    texto = ('The Breast Cancer Module is a vital component of our application, designed to assist in the early detection and diagnosis of breast cancer. This module enables healthcare professionals to input essential data related to breast health, including mammography results, breast tissue characteristics, family history, and other significant parameters. Utilizing a powerful machine learning algorithm, this module employs a Support Vector Machine (SVM) for classification to evaluate breast health and predict potential risks. By offering accurate and timely insights, the Breast Cancer Module equips healthcare providers to make well-informed decisions and enhance the chances of early detection and successful treatment. Our goal is to provide a valuable tool that supports medical professionals in the fight against breast cancer and ensures the best possible outcomes for patients.\n\n'
    'If you wish to assess breast health and receive valuable insights, please enter the relevant information below.')
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
    model = joblib.load('cancer_model.pkl')
    radius_mean=st.number_input('Radius mean',min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    perimeter_mean=st.number_input('Perimeter mean',min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    area_mean=st.number_input('Area mean',min_value=0.0, max_value=3000.0, value=500.0, step=0.1)
    compactness_mean=st.number_input('Compactness mean',min_value=0.00, max_value=0.90, value=0.10, step=0.01)
    concavity_mean= st.number_input('Concavity mean',min_value=0.00, max_value=0.90, value=0.10, step=0.01)
    concave_points_mean= st.number_input('concave points mean',min_value=0.00, max_value=0.90, value=0.10, step=0.01)
    radius_worst= st.number_input('Radius worst',min_value = 0.0,max_value= 50.0,value = 15.0,step=0.1)
    perimeter_worst= st.number_input('Radius worst',min_value = 0.0,max_value= 300.0,value = 100.0,step=0.1)
    area_worst = st.number_input('Area worst',min_value = 0.0,max_value= 5000.0,value = 800.0,step=0.1)
    compactness_worst= st.number_input('Compactness worst',min_value = 0.00,max_value= 3.0,value = 0.20,step=0.01)
    concavity_worst=st.number_input('Concavity worst',min_value = 0.00,max_value= 3.0,value = 0.20,step=0.01)
    concave_points_worst = st.number_input('Concave points worst',min_value = 0.00,max_value= 3.0,value = 0.20,step=0.01)
    if st.button('Predict'):
        data = {
            'radius_mean': radius_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave_points_mean': concave_points_mean,
            'radius_worst':radius_worst,
            'perimeter_worst':perimeter_worst,
            'area_worst':area_worst,
            'compactness_worst':compactness_worst,
            'concavity_worst':concavity_worst,
            'concave_points_worst':concave_points_worst
        }

        df = pd.DataFrame.from_dict([data])
        scaler = joblib.load('scaler_cancer.pkl')
        data = scaler.transform(df)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)
        print(prediction)

        if prediction[0] == 0:
            st.success('Not Breast Cancer')
        else:
            st.error('Breast Cancer')
        probabilities = model.predict_proba(input_data_reshaped)
        st.write(probabilities)
        st.write("0 = Not Breast Cancer, 1 = Breast Cancer")