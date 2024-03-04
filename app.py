import streamlit as st
import pickle
import numpy as np
import pandas as pd

columns = [	'Age','SibSp', 'Parch',	'Fare',	'Pclass_2',	'Pclass_3',	'Sex_male',	'Embarked_Q',	'Embarked_S']

# Load the model from the pickle file
with open('logistic_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)
    
# Load the model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
    

# Create the Streamlit app
st.title('Did they survive? :ship:')

# Define the user inputs
age = st.number_input("Choose age", 0, 100)
sibsp  = st.number_input("select the number of Sibling and Spous:",0, 1000)
parch  = st.number_input("select the number of Parents and Child:",0, 1000)
fare = st.number_input("Input Fare Price", 0, 1000)
pclass = st.selectbox("Choose class", ['Class_1', 'Class_2', 'Class_3'])
sex = st.selectbox("Choose sex", ['male', 'female'])
embarked = st.selectbox("Did they Embark?", ['Q', 'S', 'C'])

# Define the dropdown menu to select the model
model_name = st.selectbox("Choose the model", ['Logistic Regression', 'SVM'])

# Define the prediction function
def predict_survival():
    
    # Create a new DataFrame with the missing encoded columns
    encoded_cols = pd.DataFrame([[1 if pclass == 'Pclass_2' else 0,
                                  1 if pclass == 'Pclass_3' else 0,
                                  1 if sex == 'Sex_male' else 0,
                                  1 if embarked == 'Embarked_Q' else 0,
                                  1 if embarked == 'Embarked_S' else 0,]],
                                columns=['Pclass_2', 'Pclass_3', 'Sex_male', 'Embarked_Q', 'Embarked_S'])
    
    # Concatenate the encoded DataFrame with the existing DataFrame
    X = pd.concat([encoded_cols, pd.DataFrame([[age, sibsp, parch,fare]], columns=['Age', 'SibSp', 'Parch','Fare'])], axis=1)
    
    # Reorder the columns to match the order in the model
    X = X[columns]    
    # Use the chosen model to make a prediction
    if model_name == 'Logistic Regression':
        model = logreg_model
    else:
        model = svm_model
    
    try:
        # Make a prediction using the logistic regression model
        prediction = model.predict(X)
        
        # Display the prediction result
        if prediction[0] == 1:
            st.success('Passenger Survived :thumbsup:')
        else:
            st.error('Passenger did not Survive :thumbsdown:')
    except Exception as e:
        st.error(f"An error occurred: {e}")
    

# Create a button to make a prediction
st.button('Predict', on_click=predict_survival)

'''
To run this file use following command :

streamlit run app.py

''' 
