import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Secret Test App")
st.write("App loaded successfully with secure key!")
