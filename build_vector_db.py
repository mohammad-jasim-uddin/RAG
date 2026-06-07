import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'worldcup_summary.csv')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')


def row_to_document(row):
    return (
        f"FIFA World Cup {row.year} hosted by {row.host}. "
        f"Winner: {row.winner}. Runner-up: {row.runner_up}. "
        f"Best player: {row.best_player}. Best goalkeeper: {row.best_goalkeeper}. "
        f"Highest goal scorer: {row.top_scorer} with {row.top_scorer_goals} goals. "
        f"Notes: {row.notes}"
    )


def build_db():
    df = pd.read_csv(DATA_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    )
    collection = client.get_or_create_collection(
        name='fifa_worldcup_history',
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [str(y) for y in df['year'].tolist()]
    docs = [row_to_document(row) for row in df.itertuples(index=False)]
    metadatas = df.to_dict(orient='records')
    existing = collection.get(ids=ids)
    if existing and existing.get('ids'):
        collection.delete(ids=existing['ids'])
    collection.add(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Built vector database with {len(ids)} World Cup records at {DB_PATH}")


if __name__ == '__main__':
    build_db()
