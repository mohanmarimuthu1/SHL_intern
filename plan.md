Project Plan: SHL Conversational Assessment Recommender
Architecture Overview

User → POST /chat → Agent Layer → Retrieval Layer → Catalog Data → Response
Phase 1 — Data Collection (Catalog Scraping)
Goal: Build a structured, searchable catalog of SHL Individual Test Solutions.

Scrape https://www.shl.com/solutions/products/productcatalog/ (Individual Tests only, skip Pre-packaged Job Solutions)
Extract per-assessment: name, url, test_type, description, duration, languages, remote_testing, adaptive_irt
Save as catalog.json — this is the single source of truth; every URL returned by the API must come from here
Build embeddings for each entry using a free embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2)
Store in FAISS index for fast semantic retrieval
Phase 2 — Retrieval Layer (RAG)
Goal: Given a query derived from conversation context, return the most relevant catalog entries.

Embed the user's accumulated intent (role + constraints) into a query vector
Retrieve top-K candidates from FAISS
Re-rank or filter by hard constraints (e.g., test type, remote testing flag)
Never surface an assessment not in catalog.json
Phase 3 — Agent Logic (LLM + Prompt Engineering)
Goal: Drive the 4 conversational behaviors correctly within 8 turns.

LLM choice: Gemini 1.5 Flash (free tier) or Groq (fast, free) — justified in approach doc

Conversation state machine:

State	Condition	Action
GATHER	Insufficient context	Ask 1 targeted clarifying question
RECOMMEND	Enough context	Retrieve + return 1–10 assessments
REFINE	User adds/changes constraints	Update shortlist, don't restart
COMPARE	User asks "difference between X and Y"	Answer from catalog data only
REFUSE	Off-topic / injection attempt	Decline, redirect
Key prompt engineering rules:

System prompt includes condensed catalog (or retrieved subset) — not model's prior knowledge
Never recommend on turn 1 for vague queries
Every URL in output must be validated against catalog.json before returning
Structured output via JSON mode or function calling
Phase 4 — FastAPI Service
Endpoints:

GET /health → {"status": "ok"} (HTTP 200)
POST /chat → stateless; receives full messages history, returns structured response
Pydantic models enforce schema strictly — deviating breaks the automated evaluator.

Timeout strategy: Keep LLM calls under 25s to stay within the 30s cap. Use streaming or fast models.

Phase 5 — Testing & Evaluation
Test all 4 behaviors: clarify, recommend, refine, compare
Test guard rails: off-topic refusal, prompt injection rejection, no hallucinated URLs
Measure Recall@10 against the 10 provided conversation traces
Edge cases: vague query on turn 1, user contradicts themselves, user refuses to answer
Phase 6 — Deployment
Platform: Render (free tier) or Railway — both support FastAPI with zero config

Set up requirements.txt, Procfile or render.yaml
Pre-load FAISS index at startup (not per-request)
Health check warms up within 2 minutes (cold start tolerance)
File Structure

shl-recommender/
├── catalog/
│   ├── scraper.py          # Scrape SHL catalog
│   └── catalog.json        # Scraped data (ground truth)
├── retrieval/
│   ├── embedder.py         # Build FAISS index
│   └── retriever.py        # Query FAISS, return candidates
├── agent/
│   ├── prompts.py          # System prompt, few-shot examples
│   ├── conversation.py     # State detection + LLM call
│   └── validator.py        # URL validation against catalog
├── api/
│   ├── main.py             # FastAPI app
│   └── schemas.py          # Pydantic models
├── tests/
│   └── test_traces.py      # Replay test traces
├── requirements.txt
└── render.yaml
Tech Stack Summary
Component	Choice	Reason
LLM	Gemini 1.5 Flash or Groq (Llama 3)	Free tier, fast
Embeddings	all-MiniLM-L6-v2	Free, local, fast
Vector store	FAISS	Lightweight, no server needed
Framework	FastAPI + Pydantic	Required by spec
Deployment	Render	Free, supports cold start
Critical Risks to Guard Against
Hallucinated URLs — validate every URL before returning
Premature recommendations — enforce minimum context before recommending
Turn cap violation — track turn count; force recommendation by turn 6-7 if context is sufficient
Schema drift — Pydantic models are the contract; test schema on every response
Slow cold starts — pre-load index at startup, not lazily