"""
YouTube Video RAG Chatbot — Simple Version
--------------------------------------------
Enter a YouTube video link, hit Submit, then ask a question about it.

Run with:
    streamlit run app.py
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="centered")


# ============================================================
# Extract video ID from any YouTube URL format
# ============================================================

def extract_video_id(url_or_id: str):
    url_or_id = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


# ============================================================
# Fetch transcript
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join(chunk.text for chunk in transcript)
        return text, None
    except TranscriptsDisabled:
        return None, "no_video"
    except Exception as e:
        return None, f"DEBUG_ERROR: {type(e).__name__}: {e}"


# ============================================================
# Build vector store
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str, text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


# ============================================================
#  RAG chain
# ============================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    prompt = PromptTemplate.from_template("""
You are a helpful AI assistant answering questions about a YouTube video using its transcript.

Use the transcript excerpts below to answer the question as helpfully as possible.
If the question is broad (like "what is this video about"), summarize the overall
topic and key points based on the excerpts given, even if they don't cover everything.

Only say "I don't know" if the excerpts are clearly unrelated to the question.

Transcript excerpts:
{context}

Question:
{question}

Answer:
""")

    model = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel_chain | prompt | model | parser


# ============================================================
# UI
# ============================================================

st.title("🎥 YouTube Video Q&A")
st.write("Enter a YouTube video link, click Submit, then ask a question about it.")

video_input = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
submit = st.button("Submit")

# Run this block only when Submit is clicked
if submit:
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("❌ No video present or found. Please check the link.")
    else:
        with st.spinner("Checking video and fetching transcript..."):
            text, err = fetch_transcript(video_id)

        if err:
            st.error(f"❌ No video present or found. ({err})")
        else:
            with st.spinner("Processing transcript..."):
                vector_store = build_vector_store(video_id, text)
                chain = build_chain(vector_store)

            # Save into session_state so it survives after this run
            st.session_state.video_id = video_id
            st.session_state.chain = chain
            st.session_state.vector_store = vector_store
            st.success("✅ Video loaded successfully! You can ask a question below.")


# ============================================================
# Show video + question box only if a video has been loaded
# ============================================================

if "video_id" in st.session_state:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    question = st.text_input("Ask a question about this video")
    ask = st.button("Get Answer")

    if ask:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Debug: show what the retriever actually pulled
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}
                    )
                    retrieved_docs = retriever.invoke(question)
                    with st.expander("🔍 Debug: Retrieved context"):
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)

                    answer = st.session_state.chain.invoke(question)
                except Exception as e:
                    answer = f"Something went wrong: {e}"
            st.subheader("Answer")
            st.write(answer)
