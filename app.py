import streamlit as st
from rag_engine import get_answer

st.set_page_config(page_title="BuildBot", page_icon="🏗️")
st.title("🏗️ BuildBot")
st.subheader("Ask your construction-related questions")

question = st.text_input("🔍 Your question:")

if st.button("Get Answer") and question:
    with st.spinner("🤖 Thinking..."):
        try:
            answer = get_answer(question)
            st.success(answer)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
