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

# Pengecekan model dan tokenizers
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")

try:
    tokenizer = joblib.load(TOKENIZER_PATH)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat tokenizer: {str(e)}")

MAX_LENGTH = 17  # Harus sama dengan max_length saat training

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
    
    try:
        prediction = model.predict(text_pad)
        if prediction is None or prediction.size == 0:
            st.error("Prediksi tidak valid.")
            return "Invalid", 0.0
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        return sentiment, prediction[0][0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        return "Error", 0.0

# Informasi aplikasi
st.set_page_config(
    page_title="Amazon Prediction Sentiment", 
    page_icon="./assets/favicon.ico" 
)

st.title("Sentiment Analysis Amazon")

# Input ulasan dari pengguna
user_input = st.text_area("Masukkan ulasan aplikasi Anda di bawah ini:", "")
submit = st.button("Prediksi Sentimen")

# Hasil prediksi
if submit and user_input.strip():
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"### Hasil Prediksi")
    st.write(f"**Sentimen**: {sentiment}")