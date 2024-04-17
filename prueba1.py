import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def Algeria_app():
    st.header("Algerian Forest Fires - Prediction")
    st.write('https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset/data')
    st.markdown('---')
    st.write('#### Why is it so important to predict fires in African forests through data?')
    st.write('Predicting fires in African forests, like Algerian Forest, through data-driven methods is crucial to safeguard biodiversity, mitigate climate change, protect human livelihoods, and ensure the well-being of local communities. Data-driven predictions enable early intervention and efficient resource allocation for fire management, reducing the devastating impacts of wildfires and their far-reaching consequences, both regionally and globally.')
    model_fire = joblib.load('Algeria_fire_forest.pkl')
    st.write('#### About the dataset: Features')
    st.write('**Fine_Fuel_Moisture_Code**, is the moisture of the dead fine fuel. These fuels are found in the surface layer of the soil.')
    st.write('**Duff_Moisture_Code**, is the humidity of the mulch (Upper layer of soil formed mainly by decomposing organic matter.) It predicts how fuels located in the middle layer of the mulch burn.')
    st.write('**Drought_Code**, evaluates the moisture content of the deeper layers of the forest floor and the largest size category dead fuels.')
    st.write('**Initial_Spread_Index**, is a numerical rating of the rate of fire spread, without the influence of fuel.')
    st.write('**Buildup_Index**, indicates the amount of vegetable fuel available and provides guidance regarding the difficulty in controlling fires.')
    st.write('**Fire_Weather_Index**, represents the intensity of fire spread and can be considered as an index of fire behavior.')

    st.markdown('---')

    Temperature = st.slider('Temperature in Celsius degrees', min_value=20, max_value=42, value=20, step=1)
    Relative_Humidity = st.slider('Relative Humidity in %', min_value=20, max_value=100, value=20, step=1)
    Wind_Speed = st.slider('Wind speed in km/h', min_value=0, max_value=50, value=0, step=1)
    Rain = st.slider('Rain total day in mm', min_value=0, max_value=20, value=0, step=1)
    Fine_Fuel_Moisture_Code = st.slider('Fine Fuel Moisture Code', min_value=28, max_value=95, value=28, step=1)
    Duff_Moisture_Code = st.slider('Duff Moisture Code (DMC)', min_value=1, max_value=80, value=1, step=1)
    Drought_Code = st.slider('Drought Code (DC) index from the FWI', min_value=5, max_value=250, value=5, step=1)
    Initial_Spread_Index = st.slider('Initial Spread Index (ISI) index from the FWI', min_value=0, max_value=20, value=0, step=1)
    Buildup_Index  = st.slider('Buildup Index (BUI) index from the FWI', min_value=1, max_value=80, value=1, step=1)
    Fire_Weather_Index = st.slider('Fire Weather Index', min_value=0, max_value=35, value=0, step=1)

    st.markdown('---')

    st.button('Predict')
    data_fire = {
            'Temperature': Temperature,
            'Relative_Humidity': Relative_Humidity,
            'Wind_Speed': Wind_Speed,
            'Rain': Rain,
            'Fine_Fuel_Moisture_Code': Fine_Fuel_Moisture_Code,
            'Duff_Moisture_Code': Duff_Moisture_Code,
            'Drought_Code': Drought_Code,
            'Initial_Spread_Index': Initial_Spread_Index,
            'Buildup_Index': Buildup_Index,
            'Fire_Weather_Index': Fire_Weather_Index
        }

    df_fire = pd.DataFrame.from_dict([data_fire])

    scaler_fire = joblib.load('scaler_fire.pkl')
    data_fire = scaler_fire.transform(df_fire)


    # changing the input_data to numpy array
    input_datafire_as_numpy_array = np.asarray(data_fire)

    # reshape the array as we are predicting for one instance
    input_datafire_reshaped = input_datafire_as_numpy_array.reshape(1, -1)

    prediction_fire = model_fire.predict(input_datafire_reshaped)
    print(prediction_fire)

    if prediction_fire[0] == 0:
        st.write('Not Fire')
    else:
         st.write('Fire')
    probabilities_fire = model_fire.predict_proba(input_datafire_reshaped)
    st.write(probabilities_fire)
    st.write("0 = Not Fire, 1 = Fire")



