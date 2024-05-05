from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import streamlit as st
import tempfile

st.title("Chat with your webpage or PDF")
st.caption("This app allows you to chat with a webpage or PDF using local llama 3 and RAG")

option = st.selectbox("Select an option", ["Webpage", "PDF"])

if option == "Webpage":
    webpage_url = st.text_input("Enter webpage URL", type="default")

    if webpage_url:
        loader = WebBaseLoader(webpage_url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        def ollama_llm(question, context):
            formatted_prompt = f"Question: {question}\n\nContext: {context}"
            response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': formatted_prompt}])
            return response['message']['content']
        
        retriever = vectorstore.as_retriever()

        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def rag_chain(question):
            retrieved_docs = retriever.invoke(question)
            formatted_context = combine_docs(retrieved_docs)
            return ollama_llm(question, formatted_context)
        
        st.success(f"Loaded {webpage_url} successfully!")

        prompt = st.text_input("Ask any question about the webpage")

        if prompt:
            result = rag_chain(prompt)
            st.write(result)

elif option == "PDF":
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        def ollama_llm(question, context):
            formatted_prompt = f"Question: {question}\n\nContext: {context}"
            response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': formatted_prompt}])
            return response['message']['content']
        
        retriever = vectorstore.as_retriever()

        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def rag_chain(question):
            retrieved_docs = retriever.invoke(question)
            formatted_context = combine_docs(retrieved_docs)
            return ollama_llm(question, formatted_context)
        
        st.success("Loaded PDF successfully!")

        prompt = st.text_input("Ask any question about the PDF")

        if prompt:
            result = rag_chain(prompt)
            st.write(result)