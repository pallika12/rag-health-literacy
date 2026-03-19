import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import textstat

# Load API key
load_dotenv()

# ─────────────────────────────────────────
# STEP 1 — LOAD THE VECTOR DATABASE
# ─────────────────────────────────────────
print("Loading knowledge base...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Get all chunks for BM25 lexical search
all_docs = list(vector_store.docstore._dict.values())
tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)
print("Knowledge base loaded successfully!")

# ─────────────────────────────────────────
# STEP 2 — HYBRID SEARCH FUNCTION
# ─────────────────────────────────────────
def hybrid_search(query, k=4):
    # Vector search — finds semantically similar chunks
    vector_results = vector_store.similarity_search(query, k=k)

    # BM25 lexical search — finds exact keyword matches
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [all_docs[i] for i in top_bm25_indices]

    # Combine both results and remove duplicates
    seen = set()
    combined = []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined[:k]

# ─────────────────────────────────────────
# STEP 3 — SYSTEM PROMPTS
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
# STEP 4 — GENERATE RESPONSE FUNCTION
# ─────────────────────────────────────────
def generate_response(query, simplified=True):
    # Get relevant chunks using hybrid search
    relevant_chunks = hybrid_search(query)

    # Combine chunks into one context block
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    # Pick which system prompt to use
    system_prompt = SIMPLIFIED_PROMPT if simplified else NON_SIMPLIFIED_PROMPT

    # Build the full prompt
    full_prompt = f"""CONTEXT:
{context}

QUERY: {query}"""

    # Call ChatGPT
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    response = llm.invoke(messages)
    answer = response.content

    # Calculate readability scores
    fkgl = textstat.flesch_kincaid_grade(answer)
    smog = textstat.smog_index(answer)

    return answer, fkgl, smog

# ─────────────────────────────────────────
# STEP 5 — TEST IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is diabetes?"

    print("\n" + "="*60)
    print("NON-SIMPLIFIED RESPONSE")
    print("="*60)
    answer, fkgl, smog = generate_response(test_query, simplified=False)
    print(answer)
    print(f"\nFKGL Score: {fkgl}")
    print(f"SMOG Score: {smog}")

    print("\n" + "="*60)
    print("SIMPLIFIED RESPONSE")
    print("="*60)
    answer, fkgl, smog = generate_response(test_query, simplified=True)
    print(answer)
    print(f"\nFKGL Score: {fkgl}")
    print(f"SMOG Score: {smog}")