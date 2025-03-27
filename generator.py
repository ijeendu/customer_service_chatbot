# Generate responses to user query
from config import USER_QUERY, OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL,NUM_RESULTS, COLLECTION_NAME, CHROMA_DB_PATH
from utils import create_logger, generate_embedding,load_db_collection
from retriever import get_closest_match_from_db
import chromadb, ollama
from typing import List
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context,observe


logger = create_logger()

load_dotenv()

# #retrieve chroma db
# def load_db_collection(chroma_db_path: str,
#                        db_collection_name: str,
#                        embed_model: str) -> chromadb.Collection:
    
#     """ 
#      This function takes in a collection name and returns a chroma collection
#     """
#     try:
#         chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         research_collection = chroma_client.get_collection(
#             name=db_collection_name,
#             embedding_function=get_embeding_function(embed_model)
#             )
#     except Exception as e:
#         logger.error(f"Error connecting to chroma db: {e}")
    
#     return research_collection


# def get_closest_match_from_db(collection_name: chromadb.Collection,
#                               embeded_user_query: list,
#                               num_results: int)-> List[str]:
#     # query vectorstore directly with user query, chroma will embed automatically explore for speed?
#     results = collection_name.query(
#         query_embeddings =[embeded_user_query],
#         n_results=num_results
#     )

#     # extract context from results
#     closest_docs = results["documents"][0]

#     return closest_docs


def augument_query(query: str, context: List[str])-> str:

    #augmented_query = f"Use the following context to answer the question: {context}\n\nQuestion: {query}"
    query_context = ""

    for i, c in enumerate(context, start=1):
        query_context += f"{i}){c}\n"

    
    augmented_query = f"""
    Use the following context to answer the question:
    \n<Question/>: {query}\n
    \n<Context/>: {query_context}
    \n<Answer/>:\n: """


    return augmented_query


@observe(as_type="generation")
def generate_ollama_response(aug_query: str, ollama_chat_model: str)-> str:
    response = ollama.chat(
        model=ollama_chat_model,
        messages = [
            {"role": "user",
             "content": aug_query}
        ],
        options = {
            "seed": 42
        }
    )

    return response["message"]["content"]


@observe(as_type="generation")
def get_query_response(user_query: str, context: List[str], chat_model: str)-> str:

    # Add context to augument prompt
    logger.info(f"Augumenting prompt with context")
    aug_query = augument_query(user_query, context)
    logger.info(f"Augumented prompt: {aug_query}\n") 

    # Generate responses
    logger.info(f"Generating response to user query...")
    ollama_user_response = generate_ollama_response(aug_query, chat_model)

    return ollama_user_response


@observe
def main():
    print(langfuse_context.get_current_trace_url())
    # # Load chroma db
    collection = load_db_collection(CHROMA_DB_PATH, COLLECTION_NAME, OLLAMA_EMBEDDING_MODEL)
    query_context = get_closest_match_from_db(collection, OLLAMA_EMBEDDING_MODEL, USER_QUERY, NUM_RESULTS)
    user_response = get_query_response(
        USER_QUERY,
        query_context,
        OLLAMA_CHAT_MODEL)
    logger.info(f"Response to user query: {user_response}")

    return


if __name__ == "__main__":
    main()