# Automated Patient Education and Health Literacy Enhancement

A RAG-based AI system designed to improve health literacy by delivering 
accurate, simplified medical information to patients.

Built as part of a Bachelor of Engineering thesis at Macquarie University.

## Project Overview

This system uses Retrieval-Augmented Generation (RAG) to retrieve 
information from trusted medical sources and generate responses 
simplified to a 6th grade reading level — making health information 
accessible to patients with low health literacy.

## System Architecture

- **RAG Framework:** LangChain
- **Embedding Model:** OpenAI text-embedding-3-small
- **Vector Database:** FAISS
- **Search:** Hybrid BM25 + Vector Search
- **Generative Model:** GPT-4o-mini
- **Readability Validation:** textstat (FKGL + SMOG)
- **User Interface:** Streamlit

## Key Results

- Baseline FKGL Score: ~10.9 (high school level)
- Simplified FKGL Score: ~5.3-6.2 (6th grade level)
- Readability improvement: ~46% reduction in grade level

## How To Run

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Add your OpenAI API key to a .env file:
   OPENAI_API_KEY=your-key-here
4. Add your PDF documents to the data/ folder
5. Run the indexing pipeline:
   python indexing.py
6. Launch the app:
   streamlit run app.py

## Data Sources

- Diabetes Australia
- World Health Organization (WHO)
- NHS (UK National Health Service)
- American Diabetes Association (ADA)
- Australian Institute of Health and Welfare (AIHW)

## Project Structure

diabetes_rag_project/
├── data/              # Medical PDF documents
├── faiss_index/       # Vector database (generated)
├── app.py             # Streamlit web interface
├── indexing.py        # PDF indexing pipeline
├── rag_pipeline.py    # RAG retrieval and generation
├── requirements.txt   # Project dependencies
└── .env               # API keys (not uploaded)

## Author

Pallika Kafle
Bachelor of Engineering — Software Engineering
Macquarie University, 2025
Supervisor: Mr. Yipeng Zhou
```

---

## Step 6 — Create a requirements.txt file

This lets anyone install all your libraries in one command. Create a file called `requirements.txt` and paste this in:
```
langchain
langchain-openai
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu
openai
tiktoken
pypdf
rank-bm25
textstat
streamlit
python-dotenv