import streamlit as st
import tempfile

from utils import create_vector_store,load_vector_store, ask_question

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.write(
    "Upload a PDF and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Reset processing state when a new PDF is uploaded
    if st.session_state.last_uploaded_file != uploaded_file.name:

        st.session_state.pdf_processed = False
        st.session_state.last_uploaded_file = uploaded_file.name

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    # --------------------------------------------------
    # Process PDF
    # --------------------------------------------------

    if st.button("Process PDF", type="primary"):

        try:

            st.session_state.pdf_processed = False

            with st.spinner(
                "Processing PDF and creating embeddings..."
            ):

                create_vector_store(pdf_path)

            st.session_state.pdf_processed = True

            st.success(
                "✅ PDF processed successfully! "
                "You can now ask questions."
            )

        except Exception as e:

            st.session_state.pdf_processed = False

            st.error(
                f"❌ Error while processing PDF:\n\n{e}"
            )

# --------------------------------------------------
# Question Section
# --------------------------------------------------

if not st.session_state.pdf_processed:

    st.info(
        "📄 Please upload and process a PDF "
        "to enable question answering."
    )

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is the purpose of PYTHONPATH?",
    disabled=not st.session_state.pdf_processed
)

# --------------------------------------------------
# Get Answer Button
# --------------------------------------------------

get_answer = st.button(
    "Get Answer",
    type="primary",
    disabled=not st.session_state.pdf_processed
)

# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if get_answer:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching the PDF and generating answer..."
            ):

                db = load_vector_store()

                answer, sources = ask_question(
                    question,
                    db
                )

            # --------------------------------------------------
            # Display Answer
            # --------------------------------------------------

            st.subheader("Answer")
            st.write(answer)

            # --------------------------------------------------
            # Display Sources ONLY if answer was found
            # --------------------------------------------------

            if sources:

                st.subheader("📚 Source Chunks")

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"Source {i}"
                    ):

                        st.write(source)

        except Exception as e:

            st.error(
                f"❌ Error while generating answer:\n\n{e}"
            )