import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    )
    return client.get_or_create_collection('fifa_worldcup_history', embedding_function=embedder)


def retrieve(query: str, k: int = 5):
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=k)
    docs = result.get('documents', [[]])[0]
    metas = result.get('metadatas', [[]])[0]
    return docs, metas


def answer_question(query: str, k: int = 5):
    docs, metas = retrieve(query, k=k)
    context = "\n".join(f"- {d}" for d in docs)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI key not found. Retrieval-only answer:\n\n" + context

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a FIFA World Cup history analyst. Answer using only this context.
If data is missing or retrospective, say so clearly.

Context:
{context}

Question: {query}
"""
    response = client.chat.completions.create(
        model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
