from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List
import os
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class SentenceEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

embedding_function = SentenceEmbeddingFunction()

def extract_text_from_pdfs(files: List[UploadFile]):
    text = ""
    for file in files:
        with open(f"temp_{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
        reader = PdfReader(f"temp_{file.filename}")
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        os.remove(f"temp_{file.filename}")
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embedding=embedding_function)
    vector_store.save_local("faiss_index")

def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True)

def get_llama4_answer(query, context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set!"

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": "You're a helpful assistant that only uses the given context."},
        {"role": "user", "content": f"Answer using this context:\n\n{context}\n\nQuestion: {query}"}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7,
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content.strip()

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    raw_text = extract_text_from_pdfs(files)
    chunks = split_text_into_chunks(raw_text)
    create_vector_store(chunks)
    return {"status": "PDFs processed successfully"}

@app.post("/query")
async def query_answer(query: str = Form(...)):
    db = load_vector_store()
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = get_llama4_answer(query, context)
    return {"answer": answer}
