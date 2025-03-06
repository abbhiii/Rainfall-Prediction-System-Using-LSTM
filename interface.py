# interface.py - Streamlit app

import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rainfall_model import RainfallLSTM  # Import the model from rainfall_model.py

# Load the trained model (this will only happen once)
model = RainfallLSTM()
model.load_state_dict(torch.load('/Users/abhinavsingh/Downloads/rainfall/rainfall_forecasting_model.pth'))
model.eval()

# Load MinMaxScaler (You should use the same scaler you used during training)
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = np.array([100, 120, 80, 70, 150, 110])  # Example, replace with actual training data
scaler.fit(train_data.reshape(-1, 1))

# Define the prediction function
def predict_rainfall(input_data, scaler, model):
    # Normalize the input data
    input_data = np.array(input_data).reshape(-1, 1)
    norm_input = scaler.transform(input_data)
    
    # Prepare the input sequence for the model (reshape to batch_size, seq_len, input_size)
    norm_input = norm_input.reshape(1, len(input_data), 1)
    input_tensor = torch.tensor(norm_input, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Inverse transform the prediction to get actual rainfall value
    predicted_rainfall = scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
    
    return predicted_rainfall[0][0]

# Streamlit Interface
st.title('Rainfall Prediction System')
st.write('Enter the rainfall data for the last 6 months to predict the next month\'s rainfall.')

# User input for last 6 months of rainfall
rainfall_data = []
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

for month in months:
    rainfall = st.number_input(f"Enter rainfall for {month} (in mm):", min_value=0.0, step=0.1)
    rainfall_data.append(rainfall)

# Button for prediction
if st.button('Predict Next Month\'s Rainfall'):
    predicted_rainfall = predict_rainfall(rainfall_data, scaler, model)
    st.write(f"The predicted rainfall for the next month is: {predicted_rainfall:.2f} mm")
