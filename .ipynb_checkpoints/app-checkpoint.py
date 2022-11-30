import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('language_detector.pkl','rb'))

def predict_lan(sentence):
    input = sentence
    prediction = model.predict(input)
    return prediction

def main():
    html_temp = """"
    <div style='background-color:#025246; padding:10px'>
    <h2 style ='color:white;text-align:center;'>Language Dectection NLP Project</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    sentence = st.text_input("Type your Sentence")

    if st.button("Predict"):
        output = predict_lan(sentence)
        st.success("This Language is {}".format(output))

if __name__=='__main__':
    main()