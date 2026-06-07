# Architecture

User -> Streamlit UI -> RAG Engine -> Chroma Vector DB -> Retrieved World Cup Records -> LLM -> Answer

## Components
1. Dataset: CSV containing World Cup facts from 1930 to 2022.
2. Embeddings: sentence-transformers all-MiniLM-L6-v2.
3. Vector database: ChromaDB persistent storage.
4. LLM: OpenAI GPT model, with retrieval-only fallback.
5. Dashboard: Streamlit + Plotly.

## Data caveat
Official Golden Ball began in 1982 and Golden Glove/Yashin award began later. Earlier best player/goalkeeper fields are retrospective/historical estimates and should be labelled as such in reports.
