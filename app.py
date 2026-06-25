import os
import uuid
import shutil
import requests
import docx

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from supabase import create_client, Client
import torch
torch.set_num_threads(1)

# ----------------------------------------------------
# Load environment
# ----------------------------------------------------
load_dotenv()

app = Flask(__name__)

# ----------------------------------------------------
# Config
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_BASE_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app.config["UPLOAD_BASE_FOLDER"] = UPLOAD_BASE_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True
os.makedirs(UPLOAD_BASE_FOLDER, exist_ok=True)

# ----------------------------------------------------
# Environment variables
# ----------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_EMBED_MODEL = os.environ.get(
    "HF_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")

if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN must be set.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_docx_file(file_path):
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())

    full_text = "\n\n".join(paragraphs)
    return [Document(page_content=full_text, metadata={"source": file_path, "page": 1})]


def load_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path, "page": 1})]


def load_documents(file_path, file_extension):
    documents = []

    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_extension == "docx":
        documents = load_docx_file(file_path)
    elif file_extension == "txt":
        documents = load_txt_file(file_path)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)


# ----------------------------------------------------
# Remote embeddings via Hugging Face Inference API
# ----------------------------------------------------
def get_embeddings(texts):
    """
    Returns list of embeddings for a list of texts using Hugging Face Inference API.
    """
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    # HF can accept a single string or a list; for stability we'll do one by one
    embeddings = []
    for text in texts:
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        embedding = response.json()

        # For some models HF may return nested list [ [..] ]
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]

        embeddings.append(embedding)

    return embeddings


# ----------------------------------------------------
# Supabase vector DB helpers
# ----------------------------------------------------
def insert_chunks_to_supabase(session_id, chunks, embeddings):
    rows = []

    for chunk, embedding in zip(chunks, embeddings):
        metadata = chunk.metadata or {}
        source_path = metadata.get("source", "")
        source_filename = os.path.basename(source_path) if source_path else "unknown"
        page = metadata.get("page", 1)

        rows.append({
            "session_id": session_id,
            "source_filename": source_filename,
            "page": page,
            "content": chunk.page_content,
            "embedding": embedding
        })

    if rows:
        supabase.table("rag_documents").insert(rows).execute()


def get_indexed_files(session_id):
    result = (
        supabase
        .table("rag_documents")
        .select("source_filename")
        .eq("session_id", session_id)
        .execute()
    )

    if not result.data:
        return []

    filenames = sorted({row["source_filename"] for row in result.data if row.get("source_filename")})
    return filenames


def clear_session_documents(session_id):
    supabase.table("rag_documents").delete().eq("session_id", session_id).execute()


def query_similar_chunks(session_id, query_embedding, top_k=5):
    result = supabase.rpc(
        "match_rag_documents",
        {
            "query_embedding": query_embedding,
            "match_session_id": session_id,
            "match_count": top_k
        }
    ).execute()

    return result.data or []


# ----------------------------------------------------
# File ingestion pipeline
# ----------------------------------------------------
def ingest_file(session_id, file_path, file_extension):
    documents = load_documents(file_path, file_extension)
    if not documents:
        return 0

    chunks = split_documents(documents)
    if not chunks:
        return 0

    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(texts)

    insert_chunks_to_supabase(session_id, chunks, embeddings)
    return len(chunks)


# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/files", methods=["GET"])
def get_files():
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        return jsonify({"success": False, "error": "Missing X-Session-ID header"}), 400

    files = get_indexed_files(session_id)
    return jsonify({"success": True, "files": files})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    session_id = (
        request.headers.get("X-Session-ID")
        or (request.get_json(silent=True) or {}).get("session_id")
        or request.form.get("session_id")
    )

    if not session_id:
        return jsonify({"success": False, "error": "Missing session ID"}), 400

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not supported. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    try:
        user_upload_dir = os.path.join(app.config["UPLOAD_BASE_FOLDER"], secure_filename(session_id))
        os.makedirs(user_upload_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(user_upload_dir, filename)
        file.save(file_path)

        file_ext = filename.rsplit(".", 1)[1].lower()
        chunks_added = ingest_file(session_id, file_path, file_ext)

        return jsonify({
            "success": True,
            "filename": filename,
            "chunks": chunks_added,
            "message": f"Successfully uploaded and indexed '{filename}' ({chunks_added} chunks)."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error indexing file: {str(e)}"}), 500


@app.route("/api/query", methods=["POST"])
def query_rag():
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        return jsonify({"success": False, "error": "Missing X-Session-ID header"}), 400

    data = request.json or {}
    question = data.get("question")
    show_references = data.get("show_references", False)

    if not question or not question.strip():
        return jsonify({"success": False, "error": "Question cannot be empty"}), 400

    try:
        indexed_files = get_indexed_files(session_id)
        if not indexed_files:
            return jsonify({
                "success": True,
                "answer": "Please upload and index some documents first before asking questions.",
                "references": []
            })

        query_embedding = get_embeddings([question])[0]
        retrieved_docs = query_similar_chunks(session_id, query_embedding, top_k=5)

        context = "\n\n".join([
            f"Source: {doc['source_filename']} (Page {doc['page']})\nContent: {doc['content']}"
            for doc in retrieved_docs
        ])

        prompt = f"""Use the following retrieved context segments from documents to answer the question.
If the context doesn't contain enough information to answer, state that you cannot find the answer in the uploaded files.
Be concise, factually accurate, and prioritize information found in the context.

Context:
{context}

Question:
{question}

Answer:"""

        if not GROQ_API_KEY:
            return jsonify({"success": False, "error": "GROQ_API_KEY is not configured"}), 500

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024
        )

        response = llm.invoke(prompt)
        answer = response.content

        references = []
        if show_references:
            references = [
                {
                    "id": doc["id"],
                    "text": doc["content"],
                    "source": doc["source_filename"],
                    "page": doc["page"],
                    "similarity": round(float(doc["similarity"]), 4)
                }
                for doc in retrieved_docs
            ]

        return jsonify({
            "success": True,
            "answer": answer,
            "references": references
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error running query: {str(e)}"}), 500


@app.route("/api/clear", methods=["POST"])
def clear_db():
    session_id = (
        request.headers.get("X-Session-ID")
        or (request.get_json(silent=True) or {}).get("session_id")
        or request.form.get("session_id")
    )

    if not session_id:
        return jsonify({"success": False, "error": "Missing session ID"}), 400

    try:
        clear_session_documents(session_id)

        user_upload_dir = os.path.join(app.config["UPLOAD_BASE_FOLDER"], secure_filename(session_id))
        if os.path.exists(user_upload_dir):
            shutil.rmtree(user_upload_dir)

        return jsonify({
            "success": True,
            "message": "Database and uploaded documents cleared successfully."
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Error resetting database: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)