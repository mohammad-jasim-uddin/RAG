# FIFA World Cup RAG Assistant

.A complete Retrieval-Augmented Generation project covering the men's FIFA World Cup from 1930 to the latest completed tournament, 2022

## Features
- Structured World Cup dataset: winners, runners-up, host, best player, best goalkeeper, highest scorer, goals.
- Chroma vector database using sentence-transformers embeddings.
- LLM-powered Q&A using OpenAI or local fallback retrieval.
- Streamlit dashboard with tournament comparison.
- Approximate project budget.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/streamlit_app.py
```

## Build vector database
The Streamlit app builds the Chroma DB automatically. You can also run:
```bash
python src/build_vector_db.py
```
