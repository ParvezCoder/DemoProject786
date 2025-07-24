import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
embed_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

loader = PyPDFLoader("data.pdf")
persist_directory = "chroma_store"
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

doc_list = []
for page in pages:
  pg_split = text_splitter.split_text(page.page_content)
  for pg_sub_split in pg_split:
    metadata = {"source" : "Ai Policy",
                "page" : page.metadata["page"] + 1}
    doc_string = Document(page_content=pg_sub_split, metadata=metadata)
    doc_list.append(doc_string)

vectorstore = Chroma.from_documents(
            documents=pages,
            embedding=embed_model,
            persist_directory=persist_directory
        )

# Load retriever
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
retriever = vectorstore.as_retriever()
print(retriever)
llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
query = "tell me about pakistan"
result = vectorstore.similarity_search(query,k=5)
for doc in result:
    print(doc.page_content)