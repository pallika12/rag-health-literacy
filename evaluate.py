"""
evaluate.py  —  MediClear batch evaluation script
Runs the 30-question evaluation set and writes results to CSV.
"""

import re
import csv
import math
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import pyphen

load_dotenv()

# ─────────────────────────────────────────
# READABILITY FUNCTIONS  (pyphen-based)
# ─────────────────────────────────────────
_dic = pyphen.Pyphen(lang='en_US')

def _count_syllables(word: str) -> int:
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if not word:
        return 0
    return max(1, _dic.inserted(word).count('-') + 1)

def _preprocess(text: str) -> str:
    """Convert bullets and newlines so sentence detection works on list responses."""
    text = re.sub(r'\n\s*[-•*]\s+', '. ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def calculate_fkgl(text: str):
    """
    Returns (fkgl, words_per_sentence, syllables_per_word).
    Formula: 0.39 × wps + 11.8 × spw − 15.59
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
    wps  = num_words / num_sentences
    spw  = num_syllables / num_words
    fkgl = round(0.39 * wps + 11.8 * spw - 15.59, 2)
    return fkgl, round(wps, 2), round(spw, 2)

def calculate_smog(text: str) -> float:
    """
    SMOG with 30-sentence correction: 3 + sqrt(polysyllables × (30 / sentences))
    Returns 0.0 for texts with fewer than 3 sentences.
    """
    text = _preprocess(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 3:
        return 0.0
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    polysyllables = sum(1 for w in words if _count_syllables(w) >= 3)
    return round(3 + math.sqrt(polysyllables * (30 / len(sentences))), 2)

# ─────────────────────────────────────────
# LOAD KNOWLEDGE BASE
# ─────────────────────────────────────────
print("Loading knowledge base...")
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
all_docs          = list(vector_store.docstore._dict.values())
tokenized_corpus  = [doc.page_content.lower().split() for doc in all_docs]
bm25              = BM25Okapi(tokenized_corpus)
print("Knowledge base loaded.\n")

# ─────────────────────────────────────────
# HYBRID SEARCH
# ─────────────────────────────────────────
def hybrid_search(query: str, k: int = 4):
    vector_results   = vector_store.similarity_search(query, k=k)
    tokenized_query  = query.lower().split()
    bm25_scores      = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)),
                               key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_results = [all_docs[i] for i in top_bm25_indices]
    seen, combined = set(), []
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
FKGL = 0.39 x (words per sentence) + 11.8 x (syllables per word) - 15.59

To score 6 or below:
- Keep average words per sentence below 14
- Keep average syllables per word below 1.5

WORD RULES — most important. Choose the shortest word every time:
- "blood sugar" not "glucose"
- "heart" not "cardiovascular"
- "signs" not "symptoms"
- "check" not "monitor"
- "medicine" not "medication"
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
- "stop" not "prevent"
- "keep" not "maintain"
- "rise" not "increase"
- "fall" not "decrease"
- "help" not "assist"
- "but" not "however"
- "so" not "therefore"
- "also" not "additionally"
- "long-term" not "chronic"
- "people" not "individuals"
- "often" not "frequently"

SENTENCE RULES:
- Every sentence must be under 14 words
- Split long ideas into two short sentences

FORMAT:
- Start with: "The documents state that..."
- End with: Source: [actual source document name]
- Use bullets for symptom/sign questions
- Use short paragraphs for "what is" questions

If answer not in CONTEXT: "I could not find this in the medical documents provided." """

RETRY_TEMPLATE = """The previous response scored FKGL {fkgl:.1f} — above the target of 6.0.

Problems found:
{issues}

Rewrite the ENTIRE response. Apply these fixes:
- Split every sentence over 10 words into two sentences
- Replace every multi-syllable word using the word list in the system prompt
- Keep every bullet point under 10 words
- Do NOT add new information"""

# ─────────────────────────────────────────
# GENERATE WITH RETRY
# ─────────────────────────────────────────
def generate_response(query: str, simplified: bool = True, max_retries: int = 3):
    relevant_chunks = hybrid_search(query)
    context         = "\n\n".join([doc.page_content for doc in relevant_chunks])

    sources = []
    for doc in relevant_chunks:
        src  = doc.metadata.get("source", "Medical Document").split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page", None)
        sources.append(f"{src} (p.{page})" if page is not None else src)
    sources_text = ", ".join(dict.fromkeys(sources))

    system_prompt = SIMPLIFIED_PROMPT if simplified else NON_SIMPLIFIED_PROMPT
    full_prompt   = f"CONTEXT:\n{context}\n\nSOURCES: {sources_text}\n\nQUESTION: {query}"

    llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": full_prompt}
    ]
    attempts_log = []

    for attempt in range(max_retries):
        response         = llm.invoke(messages)
        answer           = response.content
        fkgl, wps, spw   = calculate_fkgl(answer)
        smog             = calculate_smog(answer)
        attempts_log.append({"attempt": attempt + 1, "fkgl": fkgl, "smog": smog})

        if not simplified or fkgl <= 6.0:
            return answer, fkgl, smog, attempts_log

        if attempt < max_retries - 1:
            issues = []
            if wps > 14:
                issues.append(
                    f"- Sentences too long: avg {wps:.1f} words (target < 14). "
                    f"Split every long sentence."
                )
            if spw > 1.5:
                issues.append(
                    f"- Words too complex: avg {spw:.2f} syllables (target < 1.5). "
                    f"Use shorter words from the word list."
                )
            if not issues:
                issues.append("- General complexity too high — use shorter words and sentences.")
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": RETRY_TEMPLATE.format(fkgl=fkgl, issues="\n".join(issues))
            })

    return answer, fkgl, smog, attempts_log

# ─────────────────────────────────────────
# EVALUATION QUESTIONS  (30 questions)
# ─────────────────────────────────────────
QUESTIONS = [
    # Basic knowledge
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
    # Symptoms
    "What are the symptoms of type 2 diabetes?",
    "What are the signs of prediabetes?",
    "What are the signs of low blood sugar?",
    "What are the signs of high blood sugar?",
    "What are the early warning signs of diabetes?",
    # Management
    "How is type 2 diabetes managed?",
    "How is type 1 diabetes managed?",
    "How can I check my blood sugar levels?",
    "What foods should people with diabetes eat?",
    "How does exercise help people with diabetes?",
    "What medicines are used to treat type 2 diabetes?",
    "What is an HbA1c test?",
    "How often should people with diabetes see a doctor?",
    # Complications
    "What are the long-term complications of diabetes?",
    "How does diabetes affect the kidneys?",
    "How does diabetes affect the eyes?",
    "How does diabetes affect the heart?",
    "What is diabetic ketoacidosis?",
    # Prevention and risk
    "Can type 2 diabetes be prevented?",
    "Who is at risk of developing type 2 diabetes?",
]

# ─────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────
def run_evaluation(output_csv: str = "evaluation_results.csv", max_retries: int = 3):
    results = []

    print(f"Running evaluation on {len(QUESTIONS)} questions...\n")
    print(f"{'ID':<4} {'FKGL (Non-Simp)':<18} {'FKGL (Simp)':<14} {'SMOG (Simp)':<13} {'Retries'}")
    print("-" * 65)

    for i, question in enumerate(QUESTIONS, start=1):
        # Non-simplified baseline
        non_simp_answer, non_simp_fkgl, non_simp_smog, _ = generate_response(
            question, simplified=False
        )
        # Simplified with retry
        simp_answer, simp_fkgl, simp_smog, simp_log = generate_response(
            question, simplified=True, max_retries=max_retries
        )
        retries_used = len(simp_log) - 1
        target_met   = "Yes" if simp_fkgl <= 6.0 else "No"

        results.append({
            "question_id":    i,
            "question":       question,
            "simp_answer":    simp_answer,
            "simp_fkgl":      simp_fkgl,
            "simp_smog":      simp_smog,
            "simp_retries":   retries_used,
            "target_met":     target_met,
            "non_simp_answer": non_simp_answer,
            "non_simp_fkgl":  non_simp_fkgl,
            "non_simp_smog":  non_simp_smog,
            "fkgl_improvement": round(
                ((non_simp_fkgl - simp_fkgl) / non_simp_fkgl * 100)
                if non_simp_fkgl > 0 else 0, 1
            ),
        })

        print(
            f"Q{i:<3} {non_simp_fkgl:<18.1f} {simp_fkgl:<14.1f} "
            f"{simp_smog:<13.1f} {retries_used} "
            f"{'✓' if target_met == 'Yes' else '✗'}"
        )

    # Write CSV
    fieldnames = [
        "question_id", "question",
        "simp_fkgl", "simp_smog", "simp_retries", "target_met",
        "non_simp_fkgl", "non_simp_smog", "fkgl_improvement",
        "simp_answer", "non_simp_answer"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    target_met_count = sum(1 for r in results if r["target_met"] == "Yes")
    avg_simp_fkgl    = sum(r["simp_fkgl"]    for r in results) / len(results)
    avg_non_simp_fkgl= sum(r["non_simp_fkgl"] for r in results) / len(results)
    avg_improvement  = sum(r["fkgl_improvement"] for r in results) / len(results)
    avg_retries      = sum(r["simp_retries"]  for r in results) / len(results)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total questions          : {len(QUESTIONS)}")
    print(f"Target met (FKGL ≤ 6.0) : {target_met_count}/{len(QUESTIONS)} "
          f"({target_met_count/len(QUESTIONS)*100:.1f}%)")
    print(f"Avg simplified FKGL      : {avg_simp_fkgl:.2f}")
    print(f"Avg non-simplified FKGL  : {avg_non_simp_fkgl:.2f}")
    print(f"Avg FKGL improvement     : {avg_improvement:.1f}%")
    print(f"Avg retries per question : {avg_retries:.2f}")
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    run_evaluation(output_csv="evaluation_results.csv", max_retries=3)