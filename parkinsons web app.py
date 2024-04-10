# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:34:20 2023

@author: dhruv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import streamlit as st

#loading the dataset
parkinsons_data = pd.read_csv('parkinsons.csv')

#Data pre-processing
X = parkinsons_data.drop(columns=['name','status'],axis=1)
Y = parkinsons_data['status']

#Splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 2)

#data standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
    
X_test = scaler.transform(X_test)

def stdvm(input_data):
    # SVM Model Training
    # Training the SVM model with training data
    model_SVM = svm.SVC(kernel = 'linear')

    model_SVM.fit(X_train, Y_train)

    # Accuracy score on training data
    X_train_prediction = model_SVM.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    print('Accuracy Score of Training data (SVM): ', training_data_accuracy)

    # Accuracy score on testing data
    X_test_prediction = model_SVM.predict(X_test)
    testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print('Accuracy Score of Testing data (SVM): ', testing_data_accuracy)

    #predictive system
    #input_data = (156.40500,189.39800,142.82200,0.00768,0.00005,0.00372,0.00399,0.01116,0.03995,0.34800,0.01721,0.02374,0.04310,0.05164,0.03365,17.15300,0.649554,0.686080,-4.554466,0.340176,2.856676,0.322111)

    # changing input data as numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping input data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model_SVM.predict(std_data)

    print(prediction)

    if (prediction[0] == 0):
        return 0
    else :
        return 1

def lr(input_data):
    #LR model training
    #training the LR model with training data
    
    model_LR = LogisticRegression()
    model_LR.fit(X_train, Y_train)

    Y_pred = model_LR.predict(X_test)
    
    # Accuracy score on training data
    X_train_prediction = model_LR.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    print('Accuracy Score of Training data (LR): ', training_data_accuracy)

    # Accuracy score on testing data
    X_test_prediction = model_LR.predict(X_test)
    testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print('Accuracy Score of Testing data : ', testing_data_accuracy)
    
    #input_data = (156.40500,189.39800,142.82200,0.00768,0.00005,0.00372,0.00399,0.01116,0.03995,0.34800,0.01721,0.02374,0.04310,0.05164,0.03365,17.15300,0.649554,0.686080,-4.554466,0.340176,2.856676,0.322111)

    # changing input data as numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping input data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model_LR.predict(std_data)

    print(prediction)

    if (prediction[0] == 0):
        return 0
    else :
        return 1
        
def nn(input_data):
    model_NN= tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 units and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 unit and sigmoid activation
    ])

    model_NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    
    model_NN.fit(X_train, Y_train, epochs=50, batch_size=16)  # You can adjust the number of epochs and batch size

    Y_pred = model_NN.predict(X_test)
    Y_pred = (Y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    accuracy = accuracy_score(Y_test, Y_pred)
    
    print(f"Accuracy: {accuracy}")

    #input_data = (156.40500,189.39800,142.82200,0.00768,0.00005,0.00372,0.00399,0.01116,0.03995,0.34800,0.01721,0.02374,0.04310,0.05164,0.03365,17.15300,0.649554,0.686080,-4.554466,0.340176,2.856676,0.322111)

    # changing input data as numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping input data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model_NN.predict(std_data)

    print(prediction)

    if prediction[0][0] > 0.5:
        return 1
    else :
        return 0

def final_prediction(input_data):
    pred1 = stdvm(input_data)
    pred2 = lr(input_data)
    pred3 = nn(input_data)


    # Put them into a list
    binary_values = [pred1, pred2, pred3]

    # Find the value with the maximum count
    best_value = max(binary_values, key=binary_values.count)

    if best_value == 1:
        return 'The person has parkinsons disease'
    else :
        return 'The person does not have parkinsons disease'

    
    

def main():

    #GIVING A TITLE
    st.title('Parkinsons Disease Prediction')

    MDVPFoHz = st.text_input('MDVP Fo(Hz)')       
    MDVPFhiHz = st.text_input('MDVP Fhi(Hz)')     
    MDVPFloHz = st.text_input('MDVP Flo(Hz)')     
    MDVPJitter = st.text_input('MDVP Jitter(%)')    
    MDVPJitterAbs = st.text_input('MDVP Jitter(Abs)')  
    MDVPRAP = st.text_input('MDVP RAP')          
    MDVPPPQ = st.text_input('MDVP PPQ')        
    JitterDDP = st.text_input('Jitter DDP')       
    MDVPShimmer = st.text_input('MDVP Shimmer')     
    MDVPShimmerdB = st.text_input('MDVP Shimmer dB') 
    ShimmerAPQ3 = st.text_input('Shimmer APQ3')      
    ShimmerAPQ5 = st.text_input('Shimmer APQ5')      
    MDVPAPQ = st.text_input('MDVP APQ')       
    ShimmerDDA = st.text_input('Shimmer DDA')     
    NHR = st.text_input('NHR')               
    HNR = st.text_input('HNR')                         
    RPDE = st.text_input('RDPE')              
    DFA = st.text_input('DFA')             
    spread1 = st.text_input('Spread1')          
    spread2 = st.text_input('Spread2')           
    D2 = st.text_input('D2')                
    PPE = st.text_input('PPE')
    
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Parkinsons Prediction Result'):
        diagnosis = final_prediction([MDVPFoHz, MDVPFhiHz, MDVPFloHz, MDVPJitter, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmerdB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
            
    st.success(diagnosis) 
    
if __name__ == '__main__':
    main()
