import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feedback", page_icon="📣")

benefits_df = pd.read_csv("data/csv_files/benefits_data.csv")
st.dataframe(data=benefits_df)

benefit_type = st.selectbox(
    "For which benefit do you want to provide some feedback?",
    benefits_df["BenefitType"].unique().tolist(),
    index=None,
    placeholder="Select a benefit type...",
)

if benefit_type:
    benefit_subtype = st.selectbox(
    "For which subtype do you want to provide some feedback?",
    benefits_df[benefits_df["BenefitType"] == benefit_type]["BenefitSubType"].unique().tolist(),
    index=None,
    placeholder="Select a benefit subtype...",
)

if benefit_type and benefit_subtype:

    feedback = st.text_input(label="Input Benefit Feedback: ")
    if feedback:
        aux = random.choice([0, 1, 2])
        if  aux == 0:
            st.success("Positive Message")
        elif aux == 1:
            st.warning("Neutral Message")
        else:
            st.error("Negative Message")
