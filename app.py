import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure dependencies are available
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Use list comprehensions for faster filtering
    stop_words = set(stopwords.words('english'))
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Use caching to load models once, preventing re-loading on every interaction
@st.cache_resource
def load_assets():
    # Changed to 'cv' to match your CountVectorizer pickle file
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return cv, model

cv, model = load_assets()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        # Use 'cv' object to transform the text
        vector_input = cv.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")