import streamlit as st

default_page = st.Page("default_page.py", title="Main App", icon="🏠", default=True)
feedback_page = st.Page("feedback_page.py", title="Benefits' Feedback", icon="📣", default=False)
policy_page = st.Page("policy_page.py", title="Policies Q&A", icon="🤖", default=False)

pg = st.navigation([default_page, feedback_page, policy_page])
pg.run()