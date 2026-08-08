import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import  RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_huggingface import  HuggingFaceEmbeddings

from langchain_groq import ChatGroq


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please add it to your .env file."
    )


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

def create_vector_store(pdf_path):

    print("=" * 60)
    print("Step 1: Loading PDF")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(
        f"PDF Loaded Successfully. "
        f"Pages = {len(documents)}"
    )

    # --------------------------------------------------
    # Text Splitting
    # --------------------------------------------------

    print("Step 2: Splitting document")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Chunks Created = {len(chunks)}"
    )

    # --------------------------------------------------
    # Generate Embeddings + FAISS
    # --------------------------------------------------

    print("Step 3: Creating FAISS vector store")

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS vector store created")

    # --------------------------------------------------
    # Save Vector Store
    # --------------------------------------------------

    print("Step 4: Saving vector store")

    os.makedirs(
        "vector_store",
        exist_ok=True
    )

    vector_db.save_local(
        "vector_store"
    )

    print("Vector store saved successfully")

    print("=" * 60)


# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

def load_vector_store():

    if not os.path.exists(
        "vector_store/index.faiss"
    ):

        raise FileNotFoundError(
            "Vector store not found. "
            "Please upload and process a PDF first."
        )

    vector_db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(
    question,
    vector_db
):

    # --------------------------------------------------
    # Similarity Search
    # --------------------------------------------------

    retrieved_docs = vector_db.similarity_search(
        question,
        k=4
    )

    # --------------------------------------------------
    # Handle No Retrieved Documents
    # --------------------------------------------------

    if not retrieved_docs:

        return (
            "I couldn't find the answer "
            "in the uploaded PDF.",
            []
        )

    # --------------------------------------------------
    # Build Context
    # --------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    # --------------------------------------------------
    # Groq LLM
    # --------------------------------------------------

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI assistant that answers questions
from an uploaded PDF.

Use ONLY the information provided in the context.

Rules:

1. Answer the question using the provided context.
2. Combine information from multiple chunks when necessary.
3. Do not use outside knowledge.
4. Do not invent or assume information.
5. If the answer is not present in the context,
   respond exactly with:

"I couldn't find the answer in the uploaded PDF."

Context:

{context}

Question:

{question}

Answer:
"""

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    response = llm.invoke(prompt)

    answer = response.content.strip()

    # --------------------------------------------------
    # Check Whether Answer Was Found
    # --------------------------------------------------

    not_found_message = (
        "I couldn't find the answer "
        "in the uploaded PDF."
    )

    if not_found_message.lower() in answer.lower():

        return (
            not_found_message,
            []
        )

    # --------------------------------------------------
    # Prepare Source Chunks
    # --------------------------------------------------

    source_chunks = [
        doc.page_content
        for doc in retrieved_docs
    ]

    # --------------------------------------------------
    # Return Answer + Sources
    # --------------------------------------------------

    return (
        answer,
        source_chunks
    )