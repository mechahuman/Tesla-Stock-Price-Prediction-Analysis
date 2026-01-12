import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# App Configuration
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title(" Tesla Stock Price Prediction using SimpleRNN")
st.markdown(
    """
    Upload a CSV file containing **Adjusted Closing Prices for the last 60 days**
    to predict **Tesla's next-day stock price**.
    """
)

# Load Training Dataset (for scaler)
@st.cache_data
def load_training_data():
    df = pd.read_csv("TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Adj Close']]

training_data = load_training_data()

# Load Model
model = load_model("simple_rnn_model.h5")

# File Upload Section
st.subheader(" Upload Last 60 Days Adjusted Close Prices")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        if 'Adj Close' not in input_df.columns:
            st.error(" CSV must contain a column named 'Adj Close'.")
        
        elif len(input_df) != 60:
            st.error(" CSV must contain exactly 60 rows.")
        
        else:
            st.success(" File uploaded successfully!")

            # Display uploaded data
            st.subheader(" Uploaded Stock Prices")
            st.dataframe(input_df)

            # Plot uploaded data
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(input_df['Adj Close'], linewidth=2)
            ax.set_title("Uploaded Adjusted Close Prices (Last 60 Days)")
            ax.set_xlabel("Days")
            ax.set_ylabel("Adjusted Close Price")
            ax.grid(True)
            st.pyplot(fig)

            # Prediction Button
            if st.button(" Predict Next-Day Price"):
                # Scale using training scaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(training_data)

                scaled_input = scaler.transform(
                    input_df[['Adj Close']]
                )

                # Reshape for LSTM
                X_input = scaled_input.reshape(1, 60, 1)

                # Predict
                prediction_scaled = model.predict(X_input)
                prediction = scaler.inverse_transform(prediction_scaled)

                st.success(
                    f" **Next-Day Tesla Stock Price:** ${prediction[0][0]:.2f}"
                )

    except Exception as e:
        st.error(f" Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:**  
    This application is for educational purposes only. Stock market predictions are
    subject to uncertainty and should not be considered financial advice.
    """
)
