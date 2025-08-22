import streamlit as st

st.set_page_config(page_title="Policies", page_icon="🤖")

st.title("PolicyGPT")

prompt = st.chat_input("Say something")
messages = st.container()
if prompt:
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write(f"Echo: {prompt}")