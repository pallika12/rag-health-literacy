import re
import math
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import pyphen

load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MediClear: Health Literacy AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp { background: #0a0f1e; }

[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #a8b8d0 !important; }

.hero-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 50%, #071020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,120,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #3b82f6; }
.hero-subtitle {
    font-size: 1rem;
    color: #6b8099;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: #3b82f6;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.input-card {
    background: #0d1526;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.response-card {
    background: #0d1526;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.response-card.simplified { border-left: 3px solid #3b82f6; }
.response-card.medical    { border-left: 3px solid #64748b; }
.response-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.response-label.simplified { color: #3b82f6; }
.response-label.medical    { color: #64748b; }
.response-text { color: #c8d8e8; font-size: 0.95rem; line-height: 1.8; }

.score-badge {
    background: #071020;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 12px;
}
.score-badge .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4a6080;
    margin-bottom: 6px;
}
.score-badge .value          { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #ffffff; line-height: 1; }
.score-badge .value.good     { color: #22c55e; }
.score-badge .value.warn     { color: #f59e0b; }
.score-badge .value.bad      { color: #ef4444; }
.score-badge .sublabel       { font-size: 0.72rem; color: #4a6080; margin-top: 4px; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 16px;
}
.status-pill.pass { background: rgba(34,197,94,0.1);  border: 1px solid rgba(34,197,94,0.3);  color: #22c55e; }
.status-pill.near { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); color: #f59e0b; }
.status-pill.fail { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #ef4444; }

.retry-banner {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: #f59e0b;
}
.history-item {
    background: #071020;
    border: 1px solid #1a2840;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.history-q     { font-size: 0.82rem; color: #8a9bb0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.history-score { font-size: 0.75rem; color: #3b82f6; margin-top: 4px; }

.section-divider { border: none; border-top: 1px solid #1e2d4a; margin: 28px 0; }

.stTextInput > div > div > input {
    background: #071020 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #c8d8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}
.stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
}
/* Question picker buttons — override the primary blue style */
[data-testid="stExpander"] .stButton > button {
    background: #071020 !important;
    color: #a8b8d0 !important;
    border: 1px solid #1e2d4a !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    padding: 8px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
[data-testid="stExpander"] .stButton > button:hover {
    background: #0d1f3c !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] {
    background: #0d1526 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# READABILITY FUNCTIONS  (pyphen-based)
# ─────────────────────────────────────────
_dic = pyphen.Pyphen(lang='en_US')

def _count_syllables(word: str) -> int:
    """Count syllables in a single word using pyphen hyphenation."""
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if not word:
        return 0
    return max(1, _dic.inserted(word).count('-') + 1)

def _preprocess(text: str) -> str:
    """
    Convert bullet points and newlines into sentence-ending punctuation
    so the sentence counter works correctly on list-style responses.
    """
    text = re.sub(r'\n\s*[-•*]\s+', '. ', text)   # bullets → sentences
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def calculate_fkgl(text: str):
    """
    Flesch-Kincaid Grade Level using the standard formula:
        FKGL = 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59

    Returns (fkgl, words_per_sentence, syllables_per_word).
    """
    text = _preprocess(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text)
                 if s.strip() and len(s.split()) > 1]
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0.0, 0.0, 0.0

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = len(words)
    if num_words == 0:
        return 0.0, 0.0, 0.0

    num_syllables = sum(_count_syllables(w) for w in words)
    wps = num_words / num_sentences
    spw = num_syllables / num_words
    fkgl = round(0.39 * wps + 11.8 * spw - 15.59, 2)
    return fkgl, round(wps, 2), round(spw, 2)

def calculate_smog(text: str) -> float:
    """
    SMOG Index with 30-sentence correction factor for short texts:
        SMOG = 3 + sqrt(polysyllables × (30 / sentences))
    Returns 0.0 if fewer than 3 sentences are found.
    """
    text = _preprocess(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = len(sentences)
    if num_sentences < 3:
        return 0.0

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    polysyllables = sum(1 for w in words if _count_syllables(w) >= 3)
    smog = 3 + math.sqrt(polysyllables * (30 / num_sentences))
    return round(smog, 2)

# ─────────────────────────────────────────
# LOAD KNOWLEDGE BASE
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
def hybrid_search(query: str, k: int = 4):
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
Your task is to act as a conversational interface for a professional
user seeking specific medical information.

You will be given a QUERY and CONTEXT chunks from trusted medical documents.

Rules:
1. Answer using ONLY the CONTEXT provided. No outside knowledge.
2. Do NOT simplify or rephrase. Use the original medical language.
3. If the answer is not in CONTEXT say:
   "I'm sorry, I could not find that in the provided medical documents."
4. Tone must be objective, formal and academic.
5. End with: Source: [name of actual source document used]"""

SIMPLIFIED_PROMPT = """You are a health educator writing patient information.

ONLY use information from the CONTEXT provided. Do not add outside knowledge.

YOUR TARGET: Flesch-Kincaid Grade Level of 6 or below.

The FKGL formula is:
FKGL = 0.39 x (words per sentence) + 11.8 x (syllables per word) - 15.59

To score 6 or below you must:
- Keep average words per sentence below 14
- Keep average syllables per word below 1.5
- This means AT LEAST 70% of your words must be 1 syllable

WORD RULES — most important factor. Choose the shortest word every time:
- "blood sugar" not "glucose"
- "heart" not "cardiovascular"
- "signs" not "symptoms"
- "check" not "monitor"
- "drugs" or "medicine" not "medication"
- "care" not "treatment"
- "high blood sugar" not "hyperglycemia"
- "low blood sugar" not "hypoglycemia"
- "harm" not "complication"
- "shots" not "injections"
- "tired" not "fatigue"
- "linked" not "associated"
- "about" not "approximately"
- "need" not "require"
- "get" not "develop"
- "give" not "provide"
- "make" not "produce"
- "stop" not "prevent"
- "keep" not "maintain"
- "rise" not "increase"
- "fall" not "decrease"
- "use" not "utilise"
- "help" not "assist"
- "but" not "however"
- "so" not "therefore"
- "also" not "additionally"
- "food" not "nutrition"
- "long-term" not "chronic"
- "people" not "individuals"
- "often" not "frequently"
- "illness" or "condition" not "disease"
- "body" not "organism"

SENTENCE RULES:
- Every sentence must be under 14 words. Count every word before you write it.
- Never connect two ideas with "which", "although", "however" in one sentence.
- Split long ideas into two short sentences.

FORMAT RULES:
- For "What is" questions: write 2 to 3 short paragraphs
- For "What are the signs/symptoms" questions: use bullet points
- For "How is it managed" questions: use a mix of both
- Start with: "The documents state that..."
- End with: Source: [actual source document name]

GOOD EXAMPLE for "What is type 2 diabetes?":
The documents state that type 2 diabetes is a long-term illness.
It affects how the body uses blood sugar. The body does not use
insulin well. Blood sugar gets too high. This can harm the heart,
eyes and kidneys over time.

There are ways to keep it in check. Good food and daily walks help.
Some people also need drugs to control their blood sugar.

Source: Diabetes Australia

GOOD EXAMPLE for "What are the signs of diabetes?":
The documents state that diabetes has a number of common signs.

- Feeling very tired each day
- Being very thirsty often
- Needing to urinate more than normal
- Blurred sight
- Slow healing cuts
- Losing weight without trying

Source: NHS UK

If the answer is not in the CONTEXT say:
"I could not find this in the medical documents provided." """

RETRY_PROMPT_TEMPLATE = """The previous response scored FKGL {fkgl:.1f} — above the target of 6.0.

The specific problems are:
{issues}

Rewrite the ENTIRE response from scratch. Apply these fixes:
- Break any sentence over 10 words into two separate sentences
- Replace every multi-syllable word using the word list in the system prompt
- Keep every bullet point under 10 words
- Do NOT add any new information — only use what was in the original context"""

# ─────────────────────────────────────────
# GENERATE RESPONSE WITH RETRY LOOP
# ─────────────────────────────────────────
def _build_sources_text(relevant_chunks: list) -> str:
    """Build a deduplicated source string with page numbers from retrieved chunks."""
    sources = []
    for doc in relevant_chunks:
        src_name = doc.metadata.get("source", "Medical Document")
        src_name = src_name.split("\\")[-1].split("/")[-1]
        page     = doc.metadata.get("page", None)
        sources.append(f"{src_name} (p.{page})" if page is not None else src_name)
    return ", ".join(list(dict.fromkeys(sources)))

def _build_retry_message(fkgl: float, wps: float, spw: float) -> str:
    """Build a diagnostic retry message explaining exactly why FKGL target was missed."""
    issues = []
    if wps > 14:
        issues.append(
            f"- Sentences are too long: average {wps:.1f} words "
            f"(must be under 14). Split every long sentence in two."
        )
    if spw > 1.5:
        issues.append(
            f"- Words are too complex: average {spw:.2f} syllables per word "
            f"(must be under 1.5). Replace multi-syllable words using the word list."
        )
    if not issues:
        issues.append("- General complexity is too high. Use shorter words and sentences.")
    return RETRY_PROMPT_TEMPLATE.format(fkgl=fkgl, issues="\n".join(issues))

def generate_response(query: str, simplified: bool = True, max_retries: int = 3):
    """
    Generate a RAG response. In simplified mode, retry up to max_retries
    times with diagnostic feedback if FKGL exceeds 6.0.
    """
    relevant_chunks = hybrid_search(query)
    context         = "\n\n".join([doc.page_content for doc in relevant_chunks])
    sources_text    = _build_sources_text(relevant_chunks)
    system_prompt   = SIMPLIFIED_PROMPT if simplified else NON_SIMPLIFIED_PROMPT

    full_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"SOURCES AVAILABLE: {sources_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"Write a professional, clear response using only the context above. "
        f"Match the format to the type of question asked."
    )

    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": full_prompt},
    ]
    attempts_log = []

    for attempt in range(max_retries):
        response        = llm.invoke(messages)
        answer          = response.content
        fkgl, wps, spw  = calculate_fkgl(answer)
        smog            = calculate_smog(answer)
        attempts_log.append({"attempt": attempt + 1, "fkgl": fkgl, "smog": smog})

        target_met = not simplified or fkgl <= 6.0
        if target_met:
            return answer, fkgl, smog, attempts_log

        has_more_attempts = attempt < max_retries - 1
        if has_more_attempts:
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": _build_retry_message(fkgl, wps, spw)})

    return answer, fkgl, smog, attempts_log

# ─────────────────────────────────────────
# EVALUATION QUESTIONS
# ─────────────────────────────────────────
EVAL_QUESTIONS = {
    "Basic Knowledge": [
        "What is diabetes?",
        "What is prediabetes?",
        "Are all types of diabetes the same?",
        "What happens when people with diabetes do not receive insulin?",
        "What causes type 1 diabetes?",
        "What causes type 2 diabetes?",
        "What is gestational diabetes?",
        "What is insulin and what does it do?",
        "How common is diabetes in Australia?",
        "What is the difference between type 1 and type 2 diabetes?",
    ],
    "Symptoms": [
        "What are the symptoms of type 2 diabetes?",
        "What are the signs of prediabetes?",
        "What are the signs of low blood sugar?",
        "What are the signs of high blood sugar?",
        "What are the early warning signs of diabetes?",
    ],
    "Management": [
        "How is type 2 diabetes managed?",
        "How is type 1 diabetes managed?",
        "How can I check my blood sugar levels?",
        "What foods should people with diabetes eat?",
        "How does exercise help people with diabetes?",
        "What medicines are used to treat type 2 diabetes?",
        "What is an HbA1c test?",
        "How often should people with diabetes see a doctor?",
    ],
    "Complications": [
        "What are the long-term complications of diabetes?",
        "How does diabetes affect the kidneys?",
        "How does diabetes affect the eyes?",
        "How does diabetes affect the heart?",
        "What is diabetic ketoacidosis?",
    ],
    "Prevention & Risk": [
        "Can type 2 diabetes be prevented?",
        "Who is at risk of developing type 2 diabetes?",
    ],
}

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
SECTION_DIVIDER = "<hr class='section-divider'>"

SLIDER_LABELS = {
    "ease":         {1: "1 — Very hard",        2: "2 — Hard",               3: "3 — Neutral", 4: "4 — Easy",      5: "5 — Very easy"},
    "trust":        {1: "1 — Not trustworthy",   2: "2 — Slightly trustworthy", 3: "3 — Neutral", 4: "4 — Trustworthy", 5: "5 — Very trustworthy"},
    "satisfaction": {1: "1 — Very unsatisfied",  2: "2 — Unsatisfied",        3: "3 — Neutral", 4: "4 — Satisfied",  5: "5 — Very satisfied"},
}

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #ffffff; margin-bottom: 4px;'>MediClear</div>
        <div style='font-size: 0.75rem; color: #3b82f6; letter-spacing: 1px; text-transform: uppercase; font-weight: 600;'>Health Literacy AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #4a6080; margin-bottom: 12px;'>About This System</div>
    <div style='font-size: 0.83rem; color: #6b8099; line-height: 1.7; margin-bottom: 24px;'>
    A RAG-based AI system that retrieves information from trusted medical sources and
    simplifies responses to a 6th grade reading level to improve patient health literacy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #4a6080; margin-bottom: 12px;'>Knowledge Sources</div>
    """, unsafe_allow_html=True)
    for source in ["Diabetes Australia", "World Health Organization", "NHS UK",
                   "American Diabetes Assoc.", "AIHW"]:
        st.markdown(
            f"<div style='font-size: 0.82rem; color: #6b8099; padding: 6px 0; "
            f"border-bottom: 1px solid #1a2840;'>{source}</div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #4a6080; margin: 24px 0 12px 0;'>How To Use</div>
    <div style='font-size: 0.82rem; color: #6b8099; line-height: 1.8;'>
    1. Type your diabetes question<br>
    2. Click Get Answer<br>
    3. View both responses side by side<br>
    4. Check readability scores
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 24px; font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #4a6080; margin-bottom: 12px;'>Readability Target</div>
    <div style='background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px; padding: 12px 14px;'>
        <div style='font-size: 0.82rem; color: #3b82f6; font-weight: 600;'>FKGL ≤ 6.0</div>
        <div style='font-size: 0.75rem; color: #4a6080; margin-top: 2px;'>6th grade reading level</div>
    </div>
    <div style='margin-top: 12px; background: rgba(59,130,246,0.05); border: 1px solid rgba(59,130,246,0.15);
    border-radius: 8px; padding: 12px 14px;'>
        <div style='font-size: 0.78rem; color: #6b8099; line-height: 1.6;'>
            Scores calculated using <strong style="color:#a8b8d0">pyphen</strong> syllable
            counting — more accurate for medical terminology than standard libraries.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown(
            "<div style='margin-top: 28px; font-size: 0.72rem; font-weight: 600; "
            "letter-spacing: 1.2px; text-transform: uppercase; color: #4a6080; "
            "margin-bottom: 12px;'>Recent Queries</div>",
            unsafe_allow_html=True
        )
        for item in reversed(st.session_state.history[-5:]):
            fkgl_val = item["simplified_fkgl"]
            color = "#22c55e" if fkgl_val <= 6.0 else "#f59e0b" if fkgl_val <= 7.0 else "#ef4444"
            st.markdown(f"""
            <div class='history-item'>
                <div class='history-q'>Q: {item['query'][:45]}...</div>
                <div class='history-score' style='color: {color};'>FKGL {fkgl_val:.1f}
                    {"· " + str(item.get("retries", 0)) + " retries" if item.get("retries", 0) > 0 else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='hero-badge'>RAG · LLM · Health Literacy</div>
    <h1 class='hero-title'>Medi<span>Clear</span></h1>
    <p class='hero-subtitle'>Automated Patient Education & Health Literacy Enhancement
    · Powered by Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
query = st.text_input(
    "Ask a diabetes question:",
    placeholder="e.g. What is type 2 diabetes? What are the symptoms of prediabetes?",
    label_visibility="visible"
)
col_retries, col_btn = st.columns([3, 1])
with col_retries:
    max_retries = st.slider(
        "Max simplification retries (simplified mode only)",
        min_value=1, max_value=5, value=3,
        help="How many times to re-prompt if FKGL > 6.0"
    )
with col_btn:
    submit = st.button("Get Answer →", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# GENERATE AND STORE
# ─────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching knowledge base and generating response..."):
            simp_answer, simp_fkgl, simp_smog, simp_log = generate_response(
                query, simplified=True, max_retries=max_retries
            )
            med_answer, med_fkgl, med_smog, _ = generate_response(
                query, simplified=False
            )

        st.session_state.current_result = {
            "query":       query,
            "simp_answer": simp_answer,
            "simp_fkgl":   simp_fkgl,
            "simp_smog":   simp_smog,
            "simp_log":    simp_log,
            "med_answer":  med_answer,
            "med_fkgl":    med_fkgl,
            "med_smog":    med_smog,
        }
        retries_used = len(simp_log) - 1
        st.session_state.history.append({
            "query":          query,
            "simplified_fkgl": simp_fkgl,
            "retries":        retries_used
        })

# ─────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────
if st.session_state.current_result:
    r = st.session_state.current_result

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #4a6080; margin-bottom: 8px;'>Your Question</div>
    <div style='font-family: DM Serif Display, serif; font-size: 1.4rem;
    color: #ffffff; margin-bottom: 28px;'>"{r['query']}"</div>
    """, unsafe_allow_html=True)

    # ── Retry banner ──────────────────────────────────────────────────────────
    retries_used = len(r["simp_log"]) - 1
    if retries_used > 0:
        attempt_summary = " → ".join(
            f"Attempt {a['attempt']}: FKGL {a['fkgl']:.1f}" for a in r["simp_log"]
        )
        st.markdown(
            f"<div class='retry-banner'>⟳ Simplification needed {retries_used} "
            f"retry(ies) — {attempt_summary}</div>",
            unsafe_allow_html=True
        )

    # ── Side-by-side responses ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='response-card simplified'>
            <div class='response-label simplified'>Simplified Response — Patient Friendly</div>
            <div class='response-text'>{r['simp_answer'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='response-card medical'>
            <div class='response-label medical'>Medical Response — Clinical Level</div>
            <div class='response-text'>{r['med_answer'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Readability scores ────────────────────────────────────────────────────
    st.markdown(SECTION_DIVIDER, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #4a6080; margin-bottom: 16px;'>Readability Analysis</div>
    """, unsafe_allow_html=True)

    col_s, col_m = st.columns(2)

    with col_s:
        fkgl_val     = r["simp_fkgl"]
        simp_color   = "good" if fkgl_val <= 6.0 else ("warn" if fkgl_val <= 7.0 else "bad")
        status_class = "pass" if fkgl_val <= 6.0 else ("near" if fkgl_val <= 7.0 else "fail")
        status_text  = "✓ Readability target met" if fkgl_val <= 6.0 else ("⚠ Near target" if fkgl_val <= 7.0 else "✗ Above target")
        st.markdown(f"""
        <div style='font-size: 0.8rem; color: #3b82f6; font-weight: 600;
        margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;'>Simplified</div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
            <div class='score-badge'>
                <div class='label'>FKGL</div>
                <div class='value {simp_color}'>{r['simp_fkgl']:.1f}</div>
                <div class='sublabel'>Target ≤ 6.0</div>
            </div>
            <div class='score-badge'>
                <div class='label'>SMOG</div>
                <div class='value'>{r['simp_smog']:.1f}</div>
                <div class='sublabel'>Grade level</div>
            </div>
        </div>
        <div class='status-pill {status_class}'>{status_text}</div>
        """, unsafe_allow_html=True)

    with col_m:
        st.markdown(f"""
        <div style='font-size: 0.8rem; color: #64748b; font-weight: 600;
        margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;'>Medical Baseline</div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
            <div class='score-badge'>
                <div class='label'>FKGL</div>
                <div class='value'>{r['med_fkgl']:.1f}</div>
                <div class='sublabel'>Baseline</div>
            </div>
            <div class='score-badge'>
                <div class='label'>SMOG</div>
                <div class='value'>{r['med_smog']:.1f}</div>
                <div class='sublabel'>Grade level</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Improvement banner ────────────────────────────────────────────────────
    med_fkgl    = r["med_fkgl"]
    improvement = ((med_fkgl - r["simp_fkgl"]) / med_fkgl * 100) if med_fkgl > 0 else 0
    st.markdown(f"""
    <div style='margin-top: 24px; background: linear-gradient(135deg,
    rgba(59,130,246,0.08), rgba(59,130,246,0.03));
    border: 1px solid rgba(59,130,246,0.2); border-radius: 12px;
    padding: 20px 24px; display: flex; align-items: center; gap: 16px;'>
        <div style='font-family: DM Serif Display, serif; font-size: 2.4rem; color: #3b82f6;'>
            {improvement:.0f}%
        </div>
        <div>
            <div style='font-size: 0.9rem; color: #c8d8e8; font-weight: 500;'>
                Readability Improvement
            </div>
            <div style='font-size: 0.8rem; color: #4a6080; margin-top: 2px;'>
                Reduction in FKGL grade level from medical baseline to simplified response
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Retry attempt breakdown (expandable) ─────────────────────────────────
    if retries_used > 0:
        with st.expander("View retry attempt breakdown"):
            for attempt in r["simp_log"]:
                attempt_fkgl  = attempt["fkgl"]
                attempt_color = "#22c55e" if attempt_fkgl <= 6.0 else ("#f59e0b" if attempt_fkgl <= 7.0 else "#ef4444")
                st.markdown(
                    f"<div style='font-size: 0.85rem; color: {attempt_color}; "
                    f"padding: 6px 0; border-bottom: 1px solid #1e2d4a;'>"
                    f"Attempt {attempt['attempt']} — "
                    f"FKGL: <strong>{attempt_fkgl:.1f}</strong> · "
                    f"SMOG: <strong>{attempt['smog']:.1f}</strong></div>",
                    unsafe_allow_html=True
                )

    # ── User Feedback ─────────────────────────────────────────────────────────
    st.markdown(SECTION_DIVIDER, unsafe_allow_html=True)

    # Initialise feedback session state
    if "feedback_step" not in st.session_state:
        st.session_state.feedback_step = 0
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}

    FEEDBACK_STEPS = [
        {"key": "medical_background",    "question": "Do you have a medical or healthcare background?",        "type": "radio",  "options": ["Yes — I work in healthcare or have medical training", "No — I am a general member of the public"]},
        {"key": "ease_of_understanding", "question": "How easy was the simplified response to understand?",    "type": "slider", "labels": SLIDER_LABELS["ease"]},
        {"key": "trustworthiness",       "question": "How accurate and trustworthy did the information feel?", "type": "slider", "labels": SLIDER_LABELS["trust"]},
        {"key": "overall_satisfaction",  "question": "How satisfied are you with the simplified response?",    "type": "slider", "labels": SLIDER_LABELS["satisfaction"]},
        {"key": "would_use_again",       "question": "Would you use this tool to look up health information?", "type": "radio",  "options": ["Yes", "Maybe", "No"]},
    ]
    total_steps = len(FEEDBACK_STEPS)
    step        = st.session_state.feedback_step

    # Step 0 — trigger button
    if step == 0:
        st.markdown("""
        <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
        text-transform: uppercase; color: #4a6080; margin-bottom: 4px;'>User Feedback</div>
        <div style='font-size: 0.82rem; color: #6b8099; margin-bottom: 16px;'>
        Help evaluate this tool by rating the response you just received.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Leave Feedback →"):
            st.session_state.feedback_step = 1
            st.session_state.feedback_data = {}
            st.rerun()

    # Steps 1–5 — one question at a time
    elif 1 <= step <= total_steps:
        current  = FEEDBACK_STEPS[step - 1]
        progress = step / total_steps

        st.markdown(f"""
        <div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
        text-transform: uppercase; color: #4a6080; margin-bottom: 12px;'>
        User Feedback — Question {step} of {total_steps}
        </div>
        """, unsafe_allow_html=True)
        st.progress(progress)

        st.markdown(f"""
        <div style='font-size: 1.1rem; color: #c8d8e8; font-weight: 500;
        margin: 20px 0 20px 0;'>{current["question"]}</div>
        """, unsafe_allow_html=True)

        if current["type"] == "radio":
            answer = st.radio(
                current["key"],
                options=current["options"],
                label_visibility="collapsed"
            )
        else:
            answer = st.select_slider(
                current["key"],
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x, labels=current["labels"]: labels[x],
                label_visibility="collapsed"
            )

        col_back, col_next = st.columns([1, 4])
        with col_back:
            if step > 1 and st.button("← Back"):
                st.session_state.feedback_step -= 1
                st.rerun()
        with col_next:
            btn_label = "Next →" if step < total_steps else "Submit Feedback →"
            if st.button(btn_label, type="primary"):
                st.session_state.feedback_data[current["key"]] = answer
                st.session_state.feedback_step += 1
                st.rerun()

    # Step 6 — save and thank you
    else:
        import csv
        import os
        from datetime import datetime

        d             = st.session_state.feedback_data
        is_medical    = "Yes" if d.get("medical_background", "").startswith("Yes") else "No"
        feedback_file = "feedback_results.csv"
        file_exists   = os.path.isfile(feedback_file)

        with open(feedback_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "question", "medical_background",
                "ease_of_understanding", "trustworthiness",
                "overall_satisfaction", "would_use_again",
                "simp_fkgl", "simp_smog", "med_fkgl", "retries_used"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question":              r["query"],
                "medical_background":    is_medical,
                "ease_of_understanding": d.get("ease_of_understanding", ""),
                "trustworthiness":       d.get("trustworthiness", ""),
                "overall_satisfaction":  d.get("overall_satisfaction", ""),
                "would_use_again":       d.get("would_use_again", ""),
                "simp_fkgl":             r["simp_fkgl"],
                "simp_smog":             r["simp_smog"],
                "med_fkgl":              r["med_fkgl"],
                "retries_used":          len(r["simp_log"]) - 1,
            })

        st.success("Thank you! Your feedback has been saved.")
        if st.button("Rate another response"):
            st.session_state.feedback_step = 0
            st.session_state.feedback_data = {}
            st.rerun()

else:
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 3rem; margin-bottom: 16px;'>🩺</div>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem;
        color: #2a3a50; margin-bottom: 8px;'>Ask your first question</div>
        <div style='font-size: 0.85rem; color: #2a3a50;'>
        Type a diabetes question above and click Get Answer</div>
    </div>
    """, unsafe_allow_html=True)