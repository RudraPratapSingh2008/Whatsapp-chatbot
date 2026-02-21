"""
WhatsApp RAG Bot - FastAPI Server
=================================
Receives WhatsApp messages, queries ChromaDB for style-matching examples
from your real chat history, and generates personalized replies via Groq.
Conversation history stored in Supabase PostgreSQL (cloud).
"""

import os
import re
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DATABASE_URL = os.getenv("DATABASE_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "..", "chroma_db"))
CHAT_FILE = os.path.join(BASE_DIR, "..", "clearedtext.txt")

COLLECTION_NAME = "global_style"
RAG_TOP_K = 8
HISTORY_LENGTH = 10
TEMPERATURE = 0.75

# ──────────────────────────────────────────────
# System Prompt  (customize this to match YOUR style)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are acting as the phone's owner. Reply to WhatsApp messages EXACTLY as they would - same tone, vocabulary, slang, length, and energy.

CRITICAL RULES:
1. You speak in Hinglish (Hindi + English mix) - casual, warm, sometimes sarcastic
2. Use short replies unless the question needs detail
3. Common words you use: arre, hnn, bhai, yaar, toh, kya, hm, ok, ni, meko, terko, h (for hai), sahi, chal, abe, ruk, pata ni
4. Use ? for engagement, ! for emphasis
5. NEVER sound like a bot - sound human, natural, sometimes lazy
6. Match the energy of the incoming message - if they're casual, be casual. If urgent, respond quickly.
7. Reply ONLY with the message text - no labels, no "Reply:", no quotes, nothing extra
8. Keep replies SHORT (1-2 lines max) unless the topic genuinely needs more detail
9. You can be funny, sarcastic, or dismissive just like a real friend
10. Use lowercase mostly, occasional caps for emphasis only
11. Don't over-explain or be overly helpful - be natural"""

# ──────────────────────────────────────────────
# Globals (initialized at startup)
# ──────────────────────────────────────────────
embedding_model: SentenceTransformer = None
chroma_client = None
chroma_collection = None
groq_client: Groq = None
db_pool: Optional[asyncpg.Pool] = None
server_ready = False  # True once all heavy init is done


# ──────────────────────────────────────────────
# Background initialization (runs AFTER port is open)
# ──────────────────────────────────────────────
async def _heavy_init():
    """Load model, connect DB, ingest data — runs as a background task
    so the HTTP port opens immediately and Render's port scanner succeeds."""
    global embedding_model, chroma_client, chroma_collection, groq_client, db_pool, server_ready

    # 1. Connect to Supabase PostgreSQL
    print("Connecting to Supabase PostgreSQL...")
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            timeout=120,
            command_timeout=120,
            server_settings={'jit': 'off'},
        )
        print("Connected to Supabase PostgreSQL!")
    except Exception as e:
        print(f"WARNING: Failed to connect to Supabase: {e}")
        print("Server will continue without database persistence (conversation history disabled)")
        db_pool = None

    # 2. Load embedding model (biggest delay)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. ChromaDB
    print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    auto_ingest_if_needed()

    # 4. Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)

    server_ready = True
    print(f"Server fully ready! Model: {GROQ_MODEL}")


# ──────────────────────────────────────────────
# PostgreSQL helpers (Supabase - conversation history)
# ──────────────────────────────────────────────
async def get_history(contact_id: str, limit: int = HISTORY_LENGTH) -> list:
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT sender, message FROM messages WHERE contact_id = $1 ORDER BY timestamp DESC LIMIT $2",
                contact_id, limit,
            )
        return [{"role": r["sender"], "content": r["message"]} for r in reversed(rows)]
    except Exception:
        return []


async def save_message(contact_id: str, contact_name: str, role: str, content: str):
    if db_pool is None:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (contact_id, contact_name, sender, message, timestamp) VALUES ($1, $2, $3, $4, $5)",
                contact_id, contact_name, role, content, datetime.now(timezone.utc),
            )
    except Exception:
        pass


# ──────────────────────────────────────────────
# RAG: query ChromaDB for style examples
# ──────────────────────────────────────────────
def query_style_examples(message: str, top_k: int = RAG_TOP_K) -> list:
    if chroma_collection is None or chroma_collection.count() == 0:
        return []

    embedding = embedding_model.encode(message).tolist()
    results = chroma_collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, chroma_collection.count()),
    )

    examples = []
    if results and results["documents"] and results["metadatas"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            examples.append({"trigger": meta.get("trigger_message", ""), "reply": doc})
    return examples


# ──────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────
def build_prompt(style_examples: list, history: list, contact_name: str, new_message: str) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if style_examples:
        style_text = (
            "Here are examples of how you (the phone owner) have replied to "
            "similar messages before. MIMIC this exact style:\n\n"
        )
        for ex in style_examples:
            style_text += f'They said: "{ex["trigger"]}"\nYou replied: "{ex["reply"]}"\n\n'
        messages.append({"role": "system", "content": style_text})

    if history:
        for msg in history:
            messages.append(
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
            )

    messages.append({"role": "user", "content": new_message})
    return messages


# ──────────────────────────────────────────────
# Groq LLM call
# ──────────────────────────────────────────────
def generate_reply(messages: list) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=256,
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# Auto-ingestion (for fresh deploys / Render)
# ──────────────────────────────────────────────
def auto_ingest_if_needed():
    """If ChromaDB is empty, ingest from clearedtext.txt automatically."""
    global chroma_collection

    try:
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        count = chroma_collection.count()
        if count > 0:
            print(f"ChromaDB: '{COLLECTION_NAME}' has {count} documents. Ready.")
            return
    except Exception:
        pass

    if not os.path.exists(CHAT_FILE):
        print(f"WARNING: No chat data at {CHAT_FILE}. Starting with empty collection.")
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        return

    print(f"Auto-ingesting from {CHAT_FILE}...")

    with open(CHAT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    pairs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"INPUT:\s*(.*?)(?:\nOUTPUT:\s*)(.*)", block, re.DOTALL)
        if match:
            inp = match.group(1).strip()
            out = match.group(2).strip()
            if inp and out and len(out) <= 300 and out not in [".", ",", "[PHONE]", "[EMAIL]"]:
                pairs.append({"input": inp, "output": out})

    if not pairs:
        print("WARNING: No valid pairs found.")
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        return

    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Global style examples from WhatsApp chat history"},
    )

    documents, metadatas, ids, embeddings = [], [], [], []
    for i, pair in enumerate(pairs):
        documents.append(pair["output"])
        metadatas.append({"trigger_message": pair["input"]})
        ids.append(f"pair_{i}")
        embeddings.append(embedding_model.encode(pair["input"]).tolist())

    chroma_collection.add(
        documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings
    )
    print(f"Auto-ingested {len(pairs)} pairs into '{COLLECTION_NAME}'!")


# ──────────────────────────────────────────────
# FastAPI app + lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client

    print("Starting WhatsApp RAG Bot Server...")

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set!")

    # Only validate env vars here — heavy work runs in background
    # so the port opens immediately for Render's health check.
    groq_client = Groq(api_key=GROQ_API_KEY)

    # Launch heavy init as a background task
    init_task = asyncio.create_task(_heavy_init())
    print("Server port is open — heavy initialization running in background...")

    yield

    # Cleanup
    init_task.cancel()
    if db_pool:
        await db_pool.close()
        print("PostgreSQL connection pool closed.")
    print("Server shutting down.")


app = FastAPI(
    title="WhatsApp RAG Bot",
    description="AI-powered WhatsApp auto-reply that sounds like you",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────
class MessageRequest(BaseModel):
    contact_id: str
    contact_name: str
    message: str


class ReplyResponse(BaseModel):
    reply: str
    style_examples_used: int
    history_length: int


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.post("/reply", response_model=ReplyResponse)
async def reply(request: MessageRequest):
    """Receive a WhatsApp message, return an AI-generated reply in your style."""
    if not server_ready:
        return ReplyResponse(
            reply="hold on bhai, server abhi start ho rha h",
            style_examples_used=0,
            history_length=0,
        )
    try:
        style_examples = query_style_examples(request.message)
        history = await get_history(request.contact_id)
        prompt_messages = build_prompt(
            style_examples, history, request.contact_name, request.message
        )
        reply_text = generate_reply(prompt_messages)

        await save_message(request.contact_id, request.contact_name, "user", request.message)
        await save_message(request.contact_id, request.contact_name, "assistant", reply_text)

        return ReplyResponse(
            reply=reply_text,
            style_examples_used=len(style_examples),
            history_length=len(history),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    if not server_ready:
        return {
            "status": "initializing",
            "database": "pending",
            "model": GROQ_MODEL,
            "collection_count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    db_ok = False
    if db_pool is not None:
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "model": GROQ_MODEL,
        "collection_count": chroma_collection.count() if chroma_collection else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/stats")
async def stats():
    total_messages = 0
    total_contacts = 0
    contacts = []
    
    if db_pool is not None:
        try:
            async with db_pool.acquire() as conn:
                total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
                total_contacts = await conn.fetchval("SELECT COUNT(DISTINCT contact_id) FROM messages")
                contacts = await conn.fetch(
                    "SELECT contact_id, contact_name, COUNT(*) as cnt FROM messages GROUP BY contact_id, contact_name"
                )
        except Exception:
            pass

    return {
        "vector_db": {
            "collection": COLLECTION_NAME,
            "document_count": chroma_collection.count() if chroma_collection else 0,
        },
        "conversations": {
            "total_messages": total_messages or 0,
            "total_contacts": total_contacts or 0,
            "contacts": [{"id": c["contact_id"], "name": c["contact_name"], "messages": c["cnt"]} for c in contacts],
        },
    }


@app.get("/contacts")
async def contacts():
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT contact_id, contact_name, COUNT(*) as cnt, MAX(timestamp) as last_msg "
                "FROM messages GROUP BY contact_id, contact_name ORDER BY last_msg DESC"
            )
        return [
            {
                "contact_id": r["contact_id"],
                "contact_name": r["contact_name"],
                "message_count": r["cnt"],
                "last_message": r["last_msg"].isoformat() if r["last_msg"] else None,
            }
            for r in rows
        ]
    except Exception:
        return []


@app.delete("/history/{contact_id}")
async def delete_history(contact_id: str):
    if db_pool is None:
        return {"deleted": 0, "contact_id": contact_id, "status": "database_unavailable"}
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute("DELETE FROM messages WHERE contact_id = $1", contact_id)
            deleted = int(result.split()[-1])
        return {"deleted": deleted, "contact_id": contact_id}
    except Exception:
        return {"deleted": 0, "contact_id": contact_id, "status": "error"}
