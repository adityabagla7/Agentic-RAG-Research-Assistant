"""
rag_pipeline.py
----------------
This file builds the "memory of documents" part of the project — this is what RAG
(Retrieval-Augmented Generation) means.

Steps (this is the RAG pipeline every AI engineer should know by heart):
  1. LOAD    -> read raw text from files
  2. SPLIT   -> break long documents into small chunks (LLMs can't read a whole book at once)
  3. EMBED   -> turn each chunk into a list of numbers (a "vector") that captures its meaning
  4. STORE   -> save those vectors in a vector database (FAISS, running locally, free)
  5. RETRIEVE-> given a question, find the chunks whose vectors are most "similar" to it
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def build_or_load_vectorstore():
    """
    Builds a FAISS vector store from the documents in data/sample_docs the first time
    this runs, then reuses the saved index on future runs (so we don't re-embed every time).

    Uses a free, local HuggingFace embedding model (all-MiniLM-L6-v2) instead of a paid
    API — it downloads once (~90MB) the first time you run this, then runs entirely on
    your own machine for free, forever, with no API key needed for this step.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR):
        print("📂 Loading existing FAISS index from disk...")
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    print("📚 Building FAISS index from documents in data/sample_docs ...")

    # 1. LOAD — read every .txt file in the folder
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # 2. SPLIT — break into ~500 character chunks with a little overlap so we don't
    #    accidentally cut a sentence in half between two chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"   -> split {len(documents)} document(s) into {len(chunks)} chunks")

    # 3 & 4. EMBED + STORE — FAISS handles both of these in one call
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print("   -> index saved to disk for next time")

    return vectorstore


def get_retriever(k: int = 3):
    """
    Returns a retriever object. Calling retriever.invoke("some question") will return
    the top `k` most relevant chunks of text to that question.
    """
    vectorstore = build_or_load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
