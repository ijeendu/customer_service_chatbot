import streamlit as st
from config import OLLAMA_EMBEDDING_MODEL, OLLAMA_CHAT_MODEL, NUM_RESULTS, COLLECTION_NAME, CHROMA_DB_PATH

from generate_with_ollama import load_db_collection, get_query_response



st.title("My Research Chatbot")

question = st.chat_input("Ask me anything about my research")

# load vectorstore
chroma_db_collection = load_db_collection(CHROMA_DB_PATH,
                    COLLECTION_NAME,
                    OLLAMA_EMBEDDING_MODEL)


# initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = []


if question:
    st.session_state.chats.append({
        "role": "user",
        "content": question
        })



    # get response
    response = get_query_response(chroma_db_collection,
                        question,
                        OLLAMA_EMBEDDING_MODEL,
                        OLLAMA_CHAT_MODEL,
                        NUM_RESULTS)


    # add response to session state
    st.session_state.chats.append({
        "role": "assistant",
        "content": response
        })


# display response as chats
for chat in st.session_state.chats:
    st.chat_message(chat["role"]).markdown(chat["content"])



# Streamlit UI
# get user query
# return response to user query
# Keep track of chat history
# Manage context
