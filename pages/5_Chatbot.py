#Step 1: Install dependencies
#!pip install langchain openai chromadb streamlit PyPDF2

#Step 2: Import Libraries
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
import io
import tempfile
from openai import OpenAI
import os

#Step 3: Streamlit UI
st.title("Chatbot with RAG - Powered by LangChain + OpenAI")
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

print(uploaded_file)

# if uploaded_file:
#     loader = PyPDFLoader(uploaded_file)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     docs = text_splitter.split_documents(documents)

if uploaded_file:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    

    OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=OpenAI.api_key)

    vectordb = Chroma.from_documents(docs, embedding=embeddings)

    

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key )
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    
    Context: {context}
    
    Question: {question}
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask your question:")
    if query:
        response = rag_chain.invoke(query)
        st.write("🤖 Answer:", response)