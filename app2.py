
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import streamlit as st


def test_model(test_sentence):
    languages = {
    'Arabic' : 0,
    'Chinese' : 1,
    'Dutch' : 2,
    'English' : 3,
    'Estonian' : 4,
    'French' : 5,
    'Hindi' : 6,
    'Indonesian' : 7,
    'Japanese' : 8,
    'Korean' : 9,
    'Latin' : 10,
    'Persian' : 11,
    'Portugese' : 12,
    'Pushto' : 13,
    'Romanian' : 14,
    'Russian' : 15,
    'Spanish' : 16,
    'Swedish' : 17,
    'Tamil' : 18,
    'Thai' : 19,
    'Turkish' : 20,
    'Urdu' : 21
    }
    ps = PorterStemmer()
    
    with open('countvec.pkl', 'rb') as f:
        cv = pickle.load(f)
    model = pickle.load(open('language_detector.pkl','rb'))
    
    rev = re.sub('^[a-zA-Z]',' ',test_sentence)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if word not in set(stopwords.words())]
    rev = ' '.join(rev)
    
    rev = cv.transform([rev]).toarray()
    
    output = model.predict(rev)[0]
    
    keys = list(languages)
    values = list(languages.values())
    position = values.index(output)
    
    output = keys[position]
    
    return output
    
def main():
    html_temp = """"
    <div style='background-color:#025246; padding:10px'>
    <h2 style ='color:white;text-align:center;'>Language Dectection NLP Project</h2>
    <h4 style = 'color:white;'>Team Members: Darsh Kumar,  Saubhagaya Sharma,  Aniket Singh</h4>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    sentence = st.text_input("Type your Sentence")

    if st.button("Predict"):
        output1 = test_model(sentence)
        st.success("This Language is {}".format(output1))

if __name__=='__main__':
    main()