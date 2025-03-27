from config import USER_QUERY, OLLAMA_EMBEDDING_MODEL, NUM_RESULTS, COLLECTION_NAME, CHROMA_DB_PATH
from utils import create_logger, load_db_collection, generate_embedding
import chromadb
from typing import List
# from langfuse.openai import OpenAI
from dotenv import load_dotenv

from langfuse.decorators import langfuse_context,observe


logger = create_logger()


load_dotenv()


# lf_client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="Ollama"
# )

@observe
def get_closest_match_from_db(db_collection: chromadb.Collection,
                              embed_model: str,
                              query: str,
                              num_results: int)-> List[str]:
    
    # generate query embedding
    embeded_query = generate_embedding(query, embed_model)
    
    # query vectorstore directly with user query, chroma will embed automatically explore for speed?
    results = db_collection.query(
        query_embeddings =[embeded_query],
        n_results=num_results
    )

    # extract context from results
    closest_docs = results["documents"][0]

    return closest_docs


@observe
def main():
    print(langfuse_context.get_current_trace_url())
    # Load chroma db
    collection = load_db_collection(CHROMA_DB_PATH, COLLECTION_NAME, OLLAMA_EMBEDDING_MODEL)
    similar_docs = get_closest_match_from_db(collection, OLLAMA_EMBEDDING_MODEL,USER_QUERY, NUM_RESULTS)
    logger.info(f"Retrieved {len(similar_docs)} similar docs: \n{similar_docs}")


if __name__ == "__main__":
    main()