import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Harsh Singh/OneDrive/Documents/Jupyer/trained_model.sav', 'rb'))

# Create a function for prediction
def heart_prediction(input_data):

    # Convert input data to a numpy array and reshape it
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person does not have a heart disease'
    else:
        return 'The person has a heart disease'


def main():
   
    st.title('Heart Disease Prediction')

    

    # Input fields for the user to fill
    age = st.text_input('Enter your age')
    sex = st.selectbox('Enter your sex', ['Male', 'Female'])
    cp = st.selectbox('Enter your chest pain type', ['Type 1', 'Type 2', 'Type 3', 'Type 4'])  # Modify as per your model's expected values
    trestbps = st.text_input('Enter your resting blood pressure')
    chol = st.text_input('Enter your cholesterol level')
    fbs = st.selectbox('Enter fasting blood sugar', ['True', 'False'])
    restecg = st.selectbox('Enter resting electrocardiographic results', ['Normal', 'Abnormal'])  # Modify based on actual options
    thalach = st.text_input('Enter your maximum heart rate')
    exang = st.selectbox('Enter exercise induced angina', ['Yes', 'No'])
    oldpeak = st.text_input('Enter your depression induced by exercise')
    slope = st.selectbox('Enter the slope of peak exercise ST segment', ['Up', 'Flat', 'Down'])  # Modify if needed
    ca = st.text_input('Enter the number of major vessels colored by fluoroscopy')
    thal = st.selectbox('Enter thalassemia', ['Normal', 'Fixed', 'Reversible'])

    # Initialize diagnosis
    diagnosis = ''

    if st.button("Heart Test Result"):
        # Convert inputs to appropriate types for the model
        try:
            age = float(age)
            trestbps = int(trestbps)
            chol = int(chol)
            thalach = int(thalach)
            oldpeak = float(oldpeak)
            ca = int(ca)

            # Encoding categorical values
            label_encoder = LabelEncoder()

            # Encode 'sex' (Male = 1, Female = 0)
            sex = 1 if sex == 'Male' else 0

            # Encode 'cp' (as an example, adjust according to the model's training)
            cp = label_encoder.fit_transform([cp])[0]

            # Encode 'fbs' (True = 1, False = 0)
            fbs = 1 if fbs == 'True' else 0

            # Encode 'restecg' (Normal = 0, Abnormal = 1)
            restecg = label_encoder.fit_transform([restecg])[0]

            # Encode 'exang' (Yes = 1, No = 0)
            exang = 1 if exang == 'Yes' else 0

            # Encode 'slope' (Up = 0, Flat = 1, Down = 2)
            slope = label_encoder.fit_transform([slope])[0]

            # Encode 'thal' (Normal = 0, Fixed = 1, Reversible = 2)
            thal = label_encoder.fit_transform([thal])[0]

            # Create the input array for prediction
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            diagnosis = heart_prediction(input_data)

        except ValueError:
            diagnosis = "Please make sure all fields are filled correctly."

    st.success(diagnosis)


if __name__ == '__main__':
    main()
