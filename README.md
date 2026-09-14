# 🎥 YouTube Video Q&A — RAG Chatbot

Ask questions about any YouTube video using its transcript — powered by LangChain, FAISS, HuggingFace Embeddings, Groq (LLaMA 3.3), and Streamlit.

---

##  What It Does

Paste any YouTube video link, and this app will:
1. Fetch the video's transcript automatically
2. Split it into chunks and build a vector store (FAISS)
3. Let you ask questions about the video in natural language
4. Retrieve the most relevant transcript chunks and generate an answer using LLaMA 3.3 (via Groq)

No manual transcript copying. No watching the entire video. Just ask.

---

## Demo

```
1. Paste YouTube URL   →   https://www.youtube.com/watch?v=...
2. Click Submit        →   Transcript fetched + Knowledge base built
3. Ask a question      →   "What is this video about?"
4. Get Answer          →   AI answers from the transcript
```

---

##  Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Transcript Fetching | youtube-transcript-api |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (local, no server needed) |
| LLM | LLaMA 3.3 70B via Groq API |
| Orchestration | LangChain LCEL (RunnableParallel, RunnablePassthrough) |

---

##  Project Structure

```
rag_project/
├── app.py              # Main Streamlit app
├── requirements.txt    # All dependencies
├── .env                # Your API keys (never commit this)
├── .env.example        # Template for .env
└── README.md
```

##  How RAG Works Here (Simple Explanation)

```
YouTube URL
    ↓
Fetch Transcript (youtube-transcript-api)
    ↓
Split into chunks (1000 chars, 200 overlap)
    ↓
Convert chunks → Embeddings (HuggingFace)
    ↓
Store in FAISS Vector DB
    ↓
User asks a Question
    ↓
Question → Embedding → Find top 8 similar chunks (Retrieval)
    ↓
Chunks + Question → Prompt → LLaMA 3.3 via Groq (Generation)
    ↓
Answer
```

---

##  Limitations

- Only works on videos that have **captions/subtitles enabled** (auto-generated captions work fine too)
- Some videos disable transcripts entirely — the app will show a clear error in that case
- Very long videos may take a few seconds longer to process on first load (embeddings are cached after that)
- Answers are based **only on the transcript** — visual content of the video is not analyzed

---

##  Requirements

```
streamlit
youtube-transcript-api
langchain
langchain-text-splitters
langchain-huggingface
langchain-groq
langchain-community
faiss-cpu
python-dotenv
sentence-transformers
```

---
