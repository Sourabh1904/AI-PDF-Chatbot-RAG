import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(pdf_path):

    try:
        print("=" * 60)
        print("Step 1: Loading PDF")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"PDF Loaded Successfully. Pages = {len(documents)}")

        print("Step 2: Splitting")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Chunks Created = {len(chunks)}")

        print("Step 3: Creating FAISS")

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("FAISS Created")

        print("Step 4: Creating Directory")

        os.makedirs("vector_store", exist_ok=True)

        print("Directory Created")

        print("Step 5: Saving")

        vector_db.save_local("vector_store")

        print("Saved Successfully")

        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    
def load_vector_store():

    if not os.path.exists("vector_store/index.faiss"):
        raise FileNotFoundError(
            "Vector store not found. Please process the PDF first."
        )

    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_question(question, vector_db):

    retrieved_docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = f"""
You are a helpful AI Assistant.

Answer ONLY from the context below.

If the answer is not available in the context, reply:
"I couldn't find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    source_chunks = [
        doc.page_content
        for doc in retrieved_docs
    ]

    return response.content, source_chunks