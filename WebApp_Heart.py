import streamlit as st
from streamlit_option_menu import option_menu
import pickle

# Loading the model
heart_data = pickle.load(open("D:\\Disease_Prediction\\Heart_Disease_Prediction\\Ensembled_Model.pkl", 'rb'))
# diab_data = pickle.load(open("D:\\Disease_Prediction\\Heart_Disease_Prediction\\Diabetes_Prediction\\LR_Diabetes_Model.pkl, 'rb'"))

# For sidebar option
with st.sidebar:
    option = option_menu('Multiple Disease Prediction',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinson Disease Prediction'], icons=['droplet-half', 'heart', 'person'], default_index=0)

if option == 'Heart Disease Prediction':
    # Title of the page
    st.title('Heart Disease Prediction Using Ensembled Learning')

    # Getting input of Feature from the User
    Age = st.number_input(label='Age')
    Sex = st.selectbox('Sex(0 = Female, 1 = male)', (0, 1))
    ChestPain = st.selectbox('ChestPain', (0, 1, 2, 3))
    restingBloodPressure = st.number_input('Resting Blood Pressure(in mmHg)')
    serumCholestrol = st.number_input('Serum Cholestrol(in mg/dl)')
    fastingBloodSugar = st.selectbox('Fasting Blood Sugar Test > 120 mg/dl (0=False, 1=True)', (0, 1))
    restingElectrocardiographicResults = st.selectbox('Resting Electrocardiographic Results', (0, 1, 2))
    maximumHeartRate = st.number_input('Maximum Heart Rate Achieved')
    exerciseInducedAngina = st.selectbox('Exercise Induced Angina(0= NO, 1=YES)', (0, 1))
    oldPeak = st.number_input('Oldpeak = ST depression induced by exercise relative to rest')
    slope = st.selectbox('The Slope of the Peak Exercise For ST Segment', (0, 1, 2))
    numberOfMajorVessels = st.selectbox("Number of Major Vessels", (0, 1, 2, 3))
    thal = st.selectbox("Thal value (0=Normal, 1=Fixed Defect, 2=Reversable)", (0, 1, 2))

    # Code for Prediction
    # Variable to store the prediction
    heart_diagno = ''

    # Button to Predict the Disease
    if st.button('Predict Heart Disease'):
        # Assigning the values to the model
        heart_predict = heart_data.predict([[Age, Sex, ChestPain, restingBloodPressure, serumCholestrol,
                                             fastingBloodSugar, restingElectrocardiographicResults,
                                             maximumHeartRate, exerciseInducedAngina, oldPeak, slope,
                                             numberOfMajorVessels, thal]])

        if heart_predict[0] == 1:
            heart_diagno = 'The Person has Heart Disease'
        else:
            heart_diagno = 'The Person does not have Heart Disease'

    st.success(heart_diagno)


if option == 'Diabetes Prediction':
    st.title('Diabetes Prediction Using ML')

    Pregnancies = st.number_input(label='Number of times Pregnant')
    Glucose = st.number_input(label="Glucose")
    BloodPressure = st.number_input(label="Blood Pressure(in mm Hg)")
    skinThickness = st.number_input(label="Skin Thickness(in mm)")
    insulin = st.number_input(label="Insulin(mUu/ml)")
    BMI = st.number_input(label="BMI")
    DiabetesPedigree = st.number_input(label='Diabetes Pedigree Function')
    Age = st.number_input(label='Age')
    #
    # diab_diagno = ''
    #
    # if st.button("Predict Diabetes"):
    #     diab_predict = diab_data.predict([[Pregnancies, Glucose, BloodPressure, skinThickness, insulin, BMI,
    #                                        DiabetesPedigree, Age]])
    #
    #     if diab_predict[0] == 1:
    #         diab_diagno = "The Person is Diabetic"
    #     else:
    #         diab_diagno = "The Person is not Diabetic"
    # st.success(diab_diagno)


if option == 'Parkinson Disease Prediction':
    st.title('Parkinson Disease Prediciton Using Ensembled Learning')
