import os
import tempfile
import streamlit as st

from utils import (
    create_vector_store,
    load_vector_store,
    ask_question
)

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.title("📄 AI PDF Chatbot (RAG)")

# ----------------------------
# Session State
# ----------------------------
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Reset when a different PDF is uploaded
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.pdf_processed = False
        st.session_state.last_uploaded_file = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF Uploaded Successfully")

    if st.button("Process PDF"):

        try:
            with st.spinner("Processing PDF... Please wait..."):
                create_vector_store(pdf_path)

            st.session_state.pdf_processed = True

            st.success("✅ PDF processed successfully!")

        except Exception as e:
            st.session_state.pdf_processed = False
            st.error(f"❌ Error while processing PDF:\n{e}")

# ----------------------------
# Ask Question
# ----------------------------
if not st.session_state.pdf_processed:
    st.info("📄 Upload and process a PDF to enable question answering.")

question = st.text_input(
    "Ask a question",
    disabled=not st.session_state.pdf_processed
)

get_answer = st.button(
    "Get Answer",
    disabled=not st.session_state.pdf_processed
)

if get_answer:

    try:
        db = load_vector_store()

        answer, sources = ask_question(question, db)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source Chunks")

        for i, source in enumerate(sources, start=1):
            with st.expander(f"Source {i}"):
                st.write(source)

    except Exception as e:
        st.error(f"❌ {e}")