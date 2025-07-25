import sys
sys.modules["sqlite3"] = __import__("pysqlite3")

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
# Load environment variables
load_dotenv()

# Initialize embedding model and language model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Streamlit UI
st.set_page_config(page_title="AI PDF QA", layout="wide")
st.title("📄 Chat with your PDF using Gemini & ChromaDB")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Save file temporarily
        with open("data.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("data.pdf")
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        doc_list = []
        for page in pages:
            pg_split = text_splitter.split_text(page.page_content)
            for pg_sub_split in pg_split:
                metadata = {"source": "User PDF", "page": page.metadata["page"] + 1}
                doc = Document(page_content=pg_sub_split, metadata=metadata)
                doc_list.append(doc)

        # Store in Chroma
        persist_directory = "chroma_store"
        # When creating Chroma
        from langchain_chroma import Chroma

        vectorstore = Chroma.from_documents(
    documents=doc_list,
    embedding=embed_model,
    persist_directory="chroma_store"
)

        
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        st.success("PDF processed successfully! Ask your question below.")

        query = st.text_input("Ask a question about your PDF")
        if query:
            with st.spinner("Searching answer..."):
                result = qa_chain({"query": query})
                st.markdown("### ✅ Answer")
                st.write(result["result"])

                st.markdown("### 📚 Sources")
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', '?')}):**")
                    st.code(doc.page_content[:500])  # show first 500 characters
