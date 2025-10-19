import streamlit as st
import pandas as pd 
import numpy as np

st.write("Hello world")

x = st.text_input("What is your favorite movie?")

st.write(f"Your favorite movie is {x}")
st.title("Mortgage application")
is_Clicked = st.button("Submit")

if is_Clicked == True:
    st.write(f"You have submitted to mention that you liked {x} movie")
else:
    st.write("Click to Submit")
    
#working with Panda
data = pd.read_csv("movies.csv")
st.write(data)



#Draw the graph
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a","b","c"])
st.bar_chart(chart_data)
st.line_chart(chart_data)

