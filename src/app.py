import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inisialisasi NLTK
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
english_stops = set(stopwords.words('english'))

# Load model dan tokenizer
MODEL_PATH = "./models/LSTM.h5.keras"
TOKENIZER_PATH = "./models/tokenizer.pkl"

model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
MAX_LENGTH = 27  # Harus sama dengan max_length saat training

# Fungsi preprocessing
def preprocess_text(text):
    # 1. Remove HTML tags
    text = text.replace('<.*?>', '')
    # 2. Remove non-alphabetical characters and numbers
    text = text.replace('[^A-Za-z]', ' ')
    # 3. Remove extra spaces between words
    text = text.replace('\s+', ' ')
    # 4. Tokenize, lemmatize, dan remove stopwords
    tokens = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in english_stops]
    return ' '.join(tokens)

# Fungsi prediksi
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    text_seq = tokenizer.texts_to_sequences([preprocessed_text])
    text_pad = pad_sequences(text_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(text_pad)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Informasi aplikasi
st.set_page_config(
    page_title="Prediksi Sentimen Amazon Store", 
    page_icon="./assets/favicon.ico" 
)

st.title("Prediksi Sentimen Aplikasi Amazon Store")
st.write("Aplikasi ini memprediksi sentimen ulasan dari pengguna menggunakan model LSTM.")

# Input ulasan dari pengguna
user_input = st.text_area("Masukkan ulasan aplikasi Anda di bawah ini:", "")
submit = st.button("Prediksi Sentimen")

# Hasil prediksi
if submit and user_input.strip():
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"### Hasil Prediksi")
    st.write(f"**Sentimen**: {sentiment}")
    st.write(f"**Confidence Score**: {confidence:.2f}")
