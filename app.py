import os
import uuid
import tempfile
from datetime import timedelta
from urllib.parse import urlparse

import requests
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pypdf import PdfReader
import docx

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from supabase import create_client, Client
import google.generativeai as genai


# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing.")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rag_secret_key_change_this")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB max upload
app.permanent_session_lifetime = timedelta(hours=8)

# -------------------------
# Clients
# -------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# File / chunk config
# -------------------------
UPLOAD_EXTENSIONS = {".pdf", ".txt", ".docx"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


# =========================================================
# Helpers
# =========================================================
def get_session_id():
    """Create/get a unique session id for this browser session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["session_id"]


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in UPLOAD_EXTENSIONS


def read_pdf(file_path):
    text_parts = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append((page_text, i))
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")
    return text_parts


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [(content, 1)]
    except Exception as e:
        raise ValueError(f"Error reading TXT: {str(e)}")


def read_docx(file_path):
    try:
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        content = "\n".join(paragraphs)
        return [(content, 1)]
    except Exception as e:
        raise ValueError(f"Error reading DOCX: {str(e)}")


def extract_text_from_file(file_path, ext):
    ext = ext.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type.")


def chunk_documents(text_pages):
    """
    text_pages: list of tuples -> [(text, page_num), ...]
    returns list of dicts:
      [{"content": "...", "page": 1}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for text, page_num in text_pages:
        if not text.strip():
            continue
        split_chunks = splitter.split_text(text)
        for ch in split_chunks:
            if ch.strip():
                chunks.append({
                    "content": ch,
                    "page": page_num
                })
    return chunks


# def get_embeddings(texts):
#     """
#     Create embeddings using Gemini via google-generativeai.
#     Returns a list of embedding vectors.
#     """
#     embeddings = []
#     for text in texts:
#         try:
#             result = genai.embed_content(
#                 model="models/text-embedding-004",
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             vector = result["embedding"]
#             embeddings.append(vector)
#         except Exception as e:
#             raise ValueError(f"Embedding error: {str(e)}")
#     return embeddings

def get_embeddings(texts):
    embeddings = []
    for i, text in enumerate(texts):
        try:
            print(f"[DEBUG] Embedding chunk {i+1}/{len(texts)}")
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            print(f"[DEBUG] Raw embedding response type: {type(result)}")
            print(f"[DEBUG] Raw embedding response: {result}")

            vector = result["embedding"]
            print(f"[DEBUG] Embedding length: {len(vector)}")

            embeddings.append(vector)
        except Exception as e:
            print(f"[DEBUG] Embedding failed: {e}")
            raise ValueError(f"Embedding error: {str(e)}")
    return embeddings

def store_chunks_in_supabase(session_id, filename, chunks, embeddings):
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append({
            "session_id": session_id,
            "source_filename": filename,
            "page": chunk["page"],
            "content": chunk["content"],
            "embedding": emb
        })

    print(f"[DEBUG] Total rows to insert: {len(rows)}")
    if rows:
        print(f"[DEBUG] First row embedding length: {len(rows[0]['embedding'])}")

    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        print(f"[DEBUG] Inserting batch {i//batch_size + 1}, size={len(batch)}")
        result = supabase.table("rag_documents").insert(batch).execute()
        print(f"[DEBUG] Supabase insert result: {result}")

# def store_chunks_in_supabase(session_id, filename, chunks, embeddings):
#     rows = []
#     for chunk, emb in zip(chunks, embeddings):
#         rows.append({
#             "session_id": session_id,
#             "source_filename": filename,
#             "page": chunk["page"],
#             "content": chunk["content"],
#             "embedding": emb
#         })

#     # Insert in batches
#     batch_size = 50
#     for i in range(0, len(rows), batch_size):
#         batch = rows[i:i + batch_size]
#         result = supabase.table("rag_documents").insert(batch).execute()
#         # if needed, result.data contains inserted rows


def search_similar_chunks(session_id, question, top_k=TOP_K):
    """
    1) embed question
    2) call Supabase RPC match function
    """
    query_embedding = get_embeddings([question])[0]

    result = supabase.rpc(
        "match_rag_documents",
        {
            "query_embedding": query_embedding,
            "match_session_id": session_id,
            "match_count": top_k
        }
    ).execute()

    if not result.data:
        return []

    return result.data


def build_context(retrieved_rows):
    """
    Convert retrieved DB rows into context text.
    """
    context_parts = []
    for row in retrieved_rows:
        filename = row.get("source_filename", "Unknown")
        page = row.get("page", 1)
        content = row.get("content", "")
        context_parts.append(f"[Source: {filename}, Page: {page}]\n{content}")

    return "\n\n".join(context_parts)


def ask_groq(question, context):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1024
    )

    prompt = f"""
You are a helpful RAG document assistant.

Use ONLY the provided context to answer the user's question.
If the answer is not present in the context, say clearly:
"I could not find that information in the uploaded documents."

Be concise, accurate, and mention source file/page when useful.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def get_indexed_files_for_session(session_id):
    """
    Get distinct uploaded files for this session.
    """
    result = supabase.table("rag_documents") \
        .select("source_filename") \
        .eq("session_id", session_id) \
        .execute()

    files = []
    seen = set()

    if result.data:
        for row in result.data:
            fname = row.get("source_filename")
            if fname and fname not in seen:
                seen.add(fname)
                files.append(fname)

    return files


def clear_session_documents(session_id):
    supabase.table("rag_documents").delete().eq("session_id", session_id).execute()


# =========================================================
# Routes
# =========================================================
@app.route("/")
def index():
    session_id = get_session_id()
    indexed_files = get_indexed_files_for_session(session_id)
    return render_template("index.html", indexed_files=indexed_files)


@app.route("/api/files", methods=["GET"])
def api_files():
    session_id = get_session_id()
    files = get_indexed_files_for_session(session_id)
    return jsonify({
        "success": True,
        "files": files
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    session_id = get_session_id()

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part found."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"success": False, "error": "Unsupported file type. Use PDF, TXT, or DOCX."}), 400

    ext = os.path.splitext(filename)[1].lower()

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Extract
        text_pages = extract_text_from_file(temp_path, ext)

        # Merge check
        full_text = "\n".join([t for t, _ in text_pages]).strip()
        if not full_text:
            return jsonify({"success": False, "error": "Could not extract readable text from this file."}), 400

        # Chunk
        chunks = chunk_documents(text_pages)
        if not chunks:
            return jsonify({"success": False, "error": "No valid chunks were generated from this file."}), 400

        # Embed
        texts = [c["content"] for c in chunks]
        embeddings = get_embeddings(texts)

        # Store in Supabase
        store_chunks_in_supabase(session_id, filename, chunks, embeddings)

        # Return updated file list
        files = get_indexed_files_for_session(session_id)

        return jsonify({
            "success": True,
            "message": f"{filename} uploaded and indexed successfully.",
            "filename": filename,
            "num_chunks": len(chunks),
            "files": files
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Error indexing file: {str(e)}"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/ask", methods=["POST"])
def ask_question():
    session_id = get_session_id()

    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"success": False, "error": "Question is required."}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"success": False, "error": "Question cannot be empty."}), 400

        files = get_indexed_files_for_session(session_id)
        if not files:
            return jsonify({"success": False, "error": "Please upload at least one document first."}), 400

        retrieved = search_similar_chunks(session_id, question, top_k=TOP_K)
        if not retrieved:
            return jsonify({
                "success": True,
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": []
            })

        context = build_context(retrieved)
        answer = ask_groq(question, context)

        sources = []
        for row in retrieved:
            sources.append({
                "filename": row.get("source_filename"),
                "page": row.get("page"),
                "similarity": row.get("similarity")
            })

        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Error answering question: {str(e)}"}), 500


@app.route("/clear", methods=["POST"])
def clear_documents():
    session_id = get_session_id()
    try:
        clear_session_documents(session_id)
        return jsonify({"success": True, "message": "All uploaded documents cleared for this session."})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error clearing documents: {str(e)}"}), 500


@app.route("/cleanup-session", methods=["POST"])
def cleanup_session():
    """
    Optional endpoint called when tab closes / page unloads.
    It clears that browser session's uploaded docs.
    """
    session_id = session.get("session_id")
    if session_id:
        try:
            clear_session_documents(session_id)
        except Exception:
            pass
        session.pop("session_id", None)

    return jsonify({"success": True})


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)