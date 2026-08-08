# 📄 AI PDF Chatbot using RAG

> An AI-powered document question-answering system that allows users to upload a PDF and ask questions using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)

## 🚀 Overview

AI PDF Chatbot is a Retrieval-Augmented Generation application that enables users to interact with PDF documents using natural language.

The application extracts PDF content, splits it into chunks, generates embeddings using HuggingFace, stores the embeddings in FAISS, retrieves relevant document chunks, and generates grounded answers using a Groq LLM.

## ✨ Features

- 📄 PDF document upload
- ✂️ Text chunking
- 🧠 HuggingFace embeddings
- 🔎 Semantic similarity search
- 🗄️ FAISS vector database
- 🤖 Groq LLM integration
- 📚 Source chunk display
- 🔐 Secure API key management
- 🖥️ Streamlit interface
- ⏳ Get Answer disabled until PDF processing completes