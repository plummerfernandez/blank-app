import streamlit as st

api_key = st.secrets["flickr"]["api_key"]
api_secret = st.secrets["flickr"]["api_secret"]


st.title("🎈 My new streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
