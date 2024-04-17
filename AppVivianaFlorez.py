import streamlit as st
from cancerApp import cancer_app
from hepatitisApp import hepatitis_app
st.markdown("<h1 style='text-align: center;'>DoctorApp</h1>", unsafe_allow_html=True)
#st.write('Student: Viviana Florez Camacho')
st.image('doctorapp.png')
st.markdown('---')
texto = ("DoctorApp is a tool developed by student Viviana Florez with the primary goal of assisting healthcare professionals in achieving quicker and more accurate diagnoses for diseases such as breast cancer and hepatitis. This powerful tool harnesses machine learning algorithms, including SVM (Support Vector Machine) and XGBOOST, to enhance a doctor's precision in their diagnostic processes. It's crucial to emphasize that DoctorApp does not intend to replace the doctor at any point; rather, it serves as a supportive tool. Its mission is to provide healthcare professionals with valuable assistance in decision-making, relying on data and patterns that may not be readily apparent. Doctors can leverage DoctorApp's machine learning capabilities to obtain a second opinion and enhance confidence in their diagnoses.\n\n"
"With DoctorApp, the aim is to reduce the time required to obtain accurate diagnoses, potentially leading to more timely and effective patient treatments. This tool exemplifies how technology and artificial intelligence can complement medical expertise and judgment, without replacing it. The combination of medical experience with the power of machine learning algorithms can make a difference in healthcare and improve the quality of life for patients.")
# Justificar y alinear el texto
st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)
with st.sidebar:
    st.image('logoie.png')
    st.write('Student: Viviana Florez Camacho')
    modelo = st.selectbox('Select an option',('Cancer','Hepatitis'))

# Mostrar una vista diferente para cada modelo seleccionado
if modelo == 'Cancer':
    cancer_app()

elif modelo == 'Hepatitis':
    hepatitis_app()

