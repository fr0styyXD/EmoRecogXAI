# Import necessary libraries
import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Function to load the trained model
@st.cache_resource
def load_ser_model():
    model = load_model('serXAI.h5')  # Replace with your model file path
    return model

# Function to preprocess audio
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCC features
    mfccs = np.mean(mfccs.T, axis=0)  # Take the mean along the time axis
    return np.array([mfccs])  # Model expects input in 2D shape

# Function to make a prediction
def predict_emotion(features, model):
    prediction = model.predict(features)
    emotion = np.argmax(prediction)  # Assuming model outputs a one-hot encoded prediction
    return emotion

# Main application function
def main():
    # Streamlit app title and description
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file, and the model will predict the emotion.")

    # Load the model
    model = load_ser_model()

    # File upload
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        # Display file details
        st.audio(uploaded_file, format="audio/wav")
        
        # Preprocess and predict emotion
        with st.spinner("Processing..."):
            features = preprocess_audio(uploaded_file)
            emotion = predict_emotion(features, model)

        # Map the model output to emotion labels
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy" , 4: "Neutral", 5: "Surprise", 6: "Sad"}  # Update based on your labels
        st.write(f"Predicted Emotion: {emotion_dict[emotion]}")

    st.write("This demo uses a pre-trained model to classify the emotion in an audio file based on its speech characteristics.")

# Run the application
if __name__ == "_main_":
    main()