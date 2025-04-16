import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for PDFs
import docx
import os

# Set up Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCi_4vA2k7JKrMQV68oUa3ONg9dJRqJcgw"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded files
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file)
    return "\n".join([page.get_text("text") for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit UI
st.title("RAG System with Gemini Flash")
st.write("Upload a document or ask a question!")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
file_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_text = extract_text_from_docx(uploaded_file)
    else:
        file_text = uploaded_file.read().decode("utf-8")
    st.write("File uploaded successfully!")

# Load and process documents
documents = [file_text] if file_text else []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents(documents)

# Convert text chunks into embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# Function to query Gemini Flash with retrieved context
def ask_gemini(query):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature":0.7,   # creativity
            "max_output_tokens":1024,
            "top_p":0.95        # diverse response
            }
        )
    return response.text

query = st.text_input("Enter your question:")
if query:
    response = ask_gemini(query)
    st.write("### Answer:")
    st.write(response)
