Enterprise Market & Competitive Intelligence Platform

An end-to-end multi-modal intelligence platform that bridges Big Data engineering, Core Machine Learning, Deep Learning, Natural Language Processing (NLP), and Generative AI to automate competitive market analysis and executive reporting.

---

Tech Stack & Architecture
* **Big Data Layer:** PySpark (distributed data ingestion, cleaning, and Parquet data lakes)
* **Core Machine Learning:** Scikit-Learn & XGBoost (metadata classification & market-shifting impact scoring)
* **Deep Learning:** PyTorch & Hugging Face Transformers (`Sentence-Transformers`) for dense vector embeddings
* **Vector Search & NLP:** ChromaDB (high-performance semantic vector store) & hybrid search matching
* **Generative AI & RAG:** LangChain, local LLMs (Ollama/Llama-3), and automated context retrieval
* **Application Architecture:** FastAPI (REST API backend) & Streamlit (interactive executive dashboard)

---

 Project Directory Structure
```text
market_intelligence_engine/
│
├── data/                    # Storage for raw datasets, parquet lakes, and vector store
├── src/
│   ├── __init__.py
│   ├── spark_pipeline.py    # Big Data distributed ingestion & cleaning
│   ├── ml_scoring.py        # XGBoost market impact classifier
│   ├── vector_store.py      # Deep learning transformer embeddings & ChromaDB
│   └── rag_engine.py        # GenAI RAG orchestrator for executive briefs
│
├── api.py                   # FastAPI backend REST service
├── app.py                   # Streamlit user interface dashboard
├── create_mock_data.py      # Utility script for generating synthetic market data
├── requirements.txt         # Project dependencies
└── .gitignore
