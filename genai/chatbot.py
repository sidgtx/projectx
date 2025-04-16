import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set up Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCi_4vA2k7JKrMQV68oUa3ONg9dJRqJcgw"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and process documents
doc_loader = TextLoader("knowledge_base.txt")  # Load your text file
documents = doc_loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Convert text chunks into embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# Function to query Gemini Flash with retrieved context
def ask_gemini(query):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

# Streamlit UI
st.title("RAG System with Gemini Flash")
st.write("Ask me anything!")

query = st.text_input("Enter your question:")
if query:
    response = ask_gemini(query)
    st.write("### Answer:")
    st.write(response)



