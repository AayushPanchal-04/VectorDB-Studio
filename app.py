"""
app.py
======
Flask REST API server for VectorDB — a faithful Python port of the
original C++ (main.cpp + httplib) backend. Every route, parameter,
and JSON response shape matches the original so the frontend needs
no behavioral changes, only a restyle.

Run:
    pip install -r requirements.txt
    python app.py
Then open:
    http://localhost:8080
"""

import os
from flask import Flask, request, jsonify, send_from_directory

from vector_engine import get_dist_fn
from database import DIMS, VectorDB, DocumentDB, chunk_text, load_demo
from ollama_client import OllamaClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=None)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


# =====================================================================
#  CORS (mirrors the cors() helper in main.cpp)
# =====================================================================

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


# =====================================================================
#  HELPERS
# =====================================================================

def parse_vec(s: str):
    if not s:
        return []
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


# =====================================================================
#  DEMO VECTOR ENDPOINTS
# =====================================================================

@app.get("/search")
def http_search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    results = [{
        "id": h["id"],
        "metadata": h["meta"],
        "category": h["cat"],
        "distance": round(h["dist"], 6),
        "embedding": [round(x, 4) for x in h["emb"]],
    } for h in out["hits"]]

    return jsonify({
        "results": results,
        "latencyUs": out["us"],
        "algo": out["algo"],
        "metric": out["metric"],
    })


@app.post("/insert")
def http_insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})
    id_ = db.insert(meta, cat, [float(x) for x in emb], get_dist_fn("cosine"))
    return jsonify({"id": id_})


@app.delete("/delete/<int:id_>")
def http_delete(id_):
    ok = db.remove(id_)
    return jsonify({"ok": ok})


@app.get("/items")
def http_items():
    items = db.all()
    return jsonify([{
        "id": v.id,
        "metadata": v.metadata,
        "category": v.category,
        "embedding": [round(x, 4) for x in v.emb],
    } for v in items])


@app.get("/benchmark")
def http_benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify({
        "bruteforceUs": b["bfUs"],
        "kdtreeUs": b["kdUs"],
        "hnswUs": b["hnswUs"],
        "itemCount": b["n"],
    })


@app.get("/hnsw-info")
def http_hnsw_info():
    gi = db.hnsw_info()
    return jsonify(gi)


# =====================================================================
#  DOCUMENT + RAG ENDPOINTS
# =====================================================================

@app.post("/doc/insert")
def http_doc_insert():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    text = (body.get("text") or "").strip()
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.delete("/doc/delete/<int:id_>")
def http_doc_delete(id_):
    ok = doc_db.remove(id_)
    return jsonify({"ok": ok})


@app.get("/doc/list")
def http_doc_list():
    docs = doc_db.all()
    out = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": d.text.count(" ") + 1,
        })
    return jsonify(out)


@app.post("/doc/search")
def http_doc_search():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    contexts = [{
        "id": item.id,
        "title": item.title,
        "distance": round(d, 4),
    } for d, item in hits]
    return jsonify({"contexts": contexts})


@app.post("/doc/ask")
def http_doc_ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = [f"[{i + 1}] {item.title}:\n{item.text}\n\n" for i, (d, item) in enumerate(hits)]
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{''.join(ctx_parts)}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    contexts = [{
        "id": item.id,
        "title": item.title,
        "text": item.text,
        "distance": round(d, 4),
    } for d, item in hits]

    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": contexts,
        "docCount": doc_db.size(),
    })


@app.get("/status")
def http_status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.get("/stats")
def http_stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# =====================================================================
#  STATIC FRONTEND
# =====================================================================

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    ollama_up = ollama.is_available()
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)
