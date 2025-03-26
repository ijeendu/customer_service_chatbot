# create embeddings and save in vectorstore using ollama models
import glob
from PyPDF2 import PdfReader
import logging
from pathlib import Path
import ollama
import chromadb
from datetime import datetime
from typing import List
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


def create_logger():

    # create logger
    logger = logging.getLogger("ollama-embeddings")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # create console handler and set level to info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.propagate = False

    return logger


logger = create_logger()


# function to get list of  pdf files
def get_docs_list(folder_path: str) -> list:
    '''  
    This function takes in a folder path of pdf files and returns a list of pdf files
    '''

    try:
        # check if folder exists
        if not Path(folder_path).exists():
            raise Exception(f"Folder {folder_path} does not exist")
    except Exception as e:
        logger.error(f"Error getting pdf files: {e}")


    # get list of pdf files
    pdf_files = glob.glob(folder_path + "/*.pdf")
    logger.info(f"Found {len(pdf_files)} pdf files")

    return pdf_files


# uses PyPDF2 to convert a pdf to text
def convert_pdf_to_text(pdf_file: str) -> str:
    '''
    This function takes in a pdf file and returns a string of the text
    '''

    text = ""
    with open(pdf_file, "rb") as file:

        logger.info(f"Converting {Path(pdf_file).stem} to text")
        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:
            
            text += page.extract_text()
        
        logger.info(f"Finished converting {Path(pdf_file).stem} file to text")
    return text


# split each doc into sections using chunking
def chunk_text(text: str, max_chars: int, overlap: int) -> list:
    '''
    This function takes in a string of text, chunks the text and 
    returns a list of strings of text corresponding to each chunk.
    '''

    chunks = []
    start = 0
    
    if max_chars > len(text):
        max_chars = len(text)

    if overlap > max_chars:
        overlap = 0
    
    if len(text) == 0:
        raise ValueError("Text is empty")

    while start < len(text):
        end = min((start + max_chars), len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap

    return chunks


#get embedding function for chroma db
def get_embeding_function(embedding_model: str) -> OllamaEmbeddingFunction:
    
    ollama_ef = OllamaEmbeddingFunction(
        model_name = embedding_model,
        url = "http://localhost:11434/api/embeddings"
    )

    return ollama_ef


# generate embeddings for text using an ollama api
def generate_embedding(input_text: str, ollama_embed_model: str) -> list:
    ''' 
    This function takes in a string of text and returns a list of embeddings
    corresponding to the input text
    '''
    
    try:
        response = ollama.embeddings(
        model = ollama_embed_model,
        prompt = input_text,
        options={
            "seed": 42 # for reproducibility
        }
    )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

    return response["embedding"]


# create new collection
def initialise_collection(chroma_db_path: str,
                          collection_name: str,
                          embeding_model: str,
                          description: str) -> chromadb.Collection:

    '''
    This function takes in a collection name and returns a chroma collection
    '''
    # create collection
    client = chromadb.PersistentClient(path=chroma_db_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embeding_function(embeding_model),
        metadata= {
                "description": description,
                "created": datetime.now().strftime("%Y-%m-%d")
                }
    )
    return collection



def create_embeddings_from_chunked_text(
        chunked_text: List[str],
        file_path: str,
        embed_model: str) -> dict:

    '''
    This function takes in a list of chunked text and returns a dictionary of doc embeddings
    '''

    docs_embedding = {}

    # create from chunked text docs and add to db 
    for c, chunk in enumerate(chunked_text):

        # create doc id
        doc_id = f"{Path(file_path).stem}-{c}"

        # embed chunked text
        embedding = generate_embedding(chunk, embed_model)

        docs_embedding[doc_id] = {
            "content": chunk,
            "embedding": embedding
        }
    
    return docs_embedding



def add_doc_embeddings_to_vectorstore(docs_embedding: dict, collection: chromadb.Collection) -> None:
    '''
    
    This function takes in a dictionary of doc embeddings and adds them to a chroma vectorstore
    '''

    # get list of doc ids to be added
    list_doc_id = list(docs_embedding.keys())

    # get embedding for doc id
    embedding_list = [docs_embedding[doc_id]["embedding"] for doc_id in list_doc_id]

    # get doc text for doc id
    doc_list = [docs_embedding[doc_id]["content"] for doc_id in list_doc_id] 

    base_file_name = list_doc_id[0][:-2]

    meta_data_list =  [{"source": base_file_name} for doc_id in list_doc_id]

    

    # add to vector db # must be a list
    try:
        collection.add(
            ids=list_doc_id,
            embeddings=embedding_list,
            documents=doc_list,
            metadatas=meta_data_list
        )

        logger.info(f"Added {base_file_name} to collection")
    except Exception as e:
        logger.error(f"Error adding {base_file_name} to collection: {e}")

    return