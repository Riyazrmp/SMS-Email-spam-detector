import nltk
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


port = PorterStemmer()
tf = pickle.load(open("vectorizer.pkl","rb"))
m  = pickle.load(open("model.pkl","rb"))
st.title("SMS/Email spam detector")
input = st.text_area("Enter the message")

def t_message(t):
    t = t.lower()
    t = nltk.word_tokenize(t)
    x = []
    for j in t:
        if j.isalnum():
            x.append(j)
    t = x[:]
    x.clear()
    for j in t:
        if j not in stopwords.words('english') and j not in string.punctuation:
            x.append(j)
    t = x[:]
    x.clear()
    for j in t:
        x.append(port.stem(j))
    
    return " ".join(x)     

if st.button("Predict"):
    t_m = t_message(input)
    v_m = tf.transform([t_m])
    r = m.predict(v_m)[0]
    if r == 1:
        st.header("SPAM!")
    else:
        st.header("Not a Spam")