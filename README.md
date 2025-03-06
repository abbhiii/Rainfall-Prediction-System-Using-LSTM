# Rainfall-Prediction-System-Using-LSTM

# Overview
This project implements a **Rainfall Prediction System** using **Long Short-Term Memory (LSTM) neural networks**. It is designed to predict **monthly rainfall** based on historical rainfall data. An **interactive interface** built with **Streamlit** allows users to input past rainfall values and obtain predictions for the next month.

# Features
- **Time-Series Forecasting:** Uses LSTM for accurate monthly rainfall predictions.
- **User-Friendly Interface:** A web-based interactive tool using **Streamlit**.
- **Preprocessed Dataset:** Normalization applied using MinMaxScaler.
- **Evaluation Metrics:** Model performance evaluated using **MSE, Precision, Recall, F1 Score, and Accuracy**.
- **PyTorch Implementation:** Leverages PyTorch for deep learning model development.


# Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Streamlit
- NumPy
- Pandas
- scikit-learn

### Clone the Repository
```bash
git clone https://github.com/yourusername/rainfall-prediction-lstm.git
cd rainfall-prediction-lstm
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

# Usage
### Training the Model
Run the script to train the LSTM model:
```bash
python train.py
```
This will train the model and save it as `rainfall_forecasting_model.pth`.

### Running the Interactive Interface
```bash
streamlit run app.py
```
This will launch the web-based interactive tool where users can input rainfall data and get predictions.

# Model Architecture
The LSTM model consists of:
- **Input Layer**: Accepts a sequence of six months of rainfall data.
- **LSTM Layer**: Captures long-term dependencies.
- **Fully Connected Layer**: Outputs the predicted rainfall.
- **ReLU Activation**: Introduces non-linearity to learn complex patterns.



# Future Improvements
- **Integrate Additional Features:** Include temperature, humidity, and wind speed for better predictions.
- **Use Transformer Models:** Explore Attention-based models like Transformer.
- **Enhance UI:** Improve interface with more visualization features.




