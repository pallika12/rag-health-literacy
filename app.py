import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import textstat

# Load API key
load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Health Assistant",
    page_icon="🩺",
    layout="centered"
)

# ─────────────────────────────────────────
# LOAD KNOWLEDGE BASE (only loads once)
# ─────────────────────────────────────────
@st.cache_resource
def load_knowledge_base():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    all_docs = list(vector_store.docstore._dict.values())
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return vector_store, all_docs, bm25

vector_store, all_docs, bm25 = load_knowledge_base()

# ─────────────────────────────────────────
# HYBRID SEARCH
# ─────────────────────────────────────────
def hybrid_search(query, k=4):
    vector_results = vector_store.similarity_search(query, k=k)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [all_docs[i] for i in top_bm25_indices]
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined[:k]

# ─────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────
NON_SIMPLIFIED_PROMPT = """You are an AI Medical Information Retriever. 
Your primary task is to act as a conversational interface for a professional 
user seeking specific medical information.

You will be given a user's QUERY and a set of CONTEXT chunks from trusted 
high-quality medical documents.

Your job is to answer the QUERY by following these rules:
1. You MUST answer using ONLY the information provided in the CONTEXT chunks.
   Do not use any of your own internal knowledge.
2. You MUST NOT simplify, rephrase, or rewrite the answer.
3. If the CONTEXT does not contain the answer, respond with:
   "I'm sorry, I could not find that specific information in the provided 
   medical documents."
4. Your tone must be objective, formal and academic.
5. Always end your response with: Source: Diabetes Australia"""

SIMPLIFIED_PROMPT = """You are an AI Health Educator helping patients with 
very low reading ability. You MUST write at a 6th grade reading level or below.

You will be given a QUERY and CONTEXT chunks from trusted medical documents.

STRICT RULES YOU MUST FOLLOW:
1. ONLY use information from the CONTEXT. Never use outside knowledge.
2. EVERY sentence must be short — maximum 15 words per sentence.
3. ONLY use simple everyday words a 10 year old would understand.
4. NEVER use medical jargon. Instead:
   - Say "blood sugar" not "glucose" or "glycated hemoglobin"
   - Say "heart disease" not "cardiovascular complications"
   - Say "nerve damage" not "neuropathy"
5. ALWAYS use bullet points to break up information.
6. NEVER start a sentence with "You should" or "You need to"
   ALWAYS say "The information says..." or "The documents state..."
7. Keep your ENTIRE response under 120 words.
8. Use ONLY simple 1-2 syllable words.
   - Say "harm" not "damage"
   - Say "high blood sugar" not "elevated glucose levels"
   - Say "get" not "develop"
9. If the CONTEXT does not contain the answer respond with:
   "I'm sorry, I could not find that in the medical documents."
10. Always end with: Source: Diabetes Australia"""

# ─────────────────────────────────────────
# GENERATE RESPONSE
# ─────────────────────────────────────────
def generate_response(query, simplified=True):
    relevant_chunks = hybrid_search(query)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    system_prompt = SIMPLIFIED_PROMPT if simplified else NON_SIMPLIFIED_PROMPT
    full_prompt = f"""CONTEXT:
{context}

QUERY: {query}"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    response = llm.invoke(messages)
    answer = response.content
    fkgl = textstat.flesch_kincaid_grade(answer)
    smog = textstat.smog_index(answer)
    return answer, fkgl, smog

# ─────────────────────────────────────────
# USER INTERFACE
# ─────────────────────────────────────────
st.title("🩺 Diabetes Health Assistant")
st.markdown("Ask any question about diabetes and get a clear, easy to understand answer based on trusted medical sources.")
st.divider()

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What is diabetes? What is prediabetes?"
)

# Response mode toggle
mode = st.radio(
    "Select response mode:",
    ["Simplified (Patient Friendly)", "Non-Simplified (Medical)"],
    horizontal=True
)

# Submit button
if st.button("Get Answer", type="primary"):
    if query.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Finding answer from medical documents..."):
            simplified = mode == "Simplified (Patient Friendly)"
            answer, fkgl, smog = generate_response(query, simplified)

        # Display response
        st.divider()
        st.subheader("📋 Response")
        st.write(answer)

        # Display readability scores
        st.divider()
        st.subheader("📊 Readability Scores")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="FKGL Score",
                value=round(fkgl, 2),
                delta="Target: ≤ 6.0" if simplified else "Baseline"
            )

        with col2:
            st.metric(
                label="SMOG Score",
                value=round(smog, 2)
            )

        # Show colour coded feedback on readability
        if simplified:
            if fkgl <= 6.0:
                st.success("✅ Readability target met! Response is at 6th grade level or below.")
            elif fkgl <= 7.0:
                st.warning("🔶 Readability is close to target. Slightly above 6th grade level.")
            else:
                st.error("❌ Readability target not met. Response is above 6th grade level.")