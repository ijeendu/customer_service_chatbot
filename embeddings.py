# create embeddings and save in vector database using ollama models
from config import *
from utils import *
from pathlib import Path
from tqdm import tqdm


# make logger
logger = create_logger()


# process input PDF files and save in chromadb vectorstore
def process_documents_and_add_to_vectorstore(pdf_files_path: str,
                                             embedding_model: str,
                                             db_folder: str,
                                             collection_name: str,
                                             coll_description: str,
                                             chunk_size: int,
                                             overlap: int) -> None:
    '''
    1. Get list of pdf files
    2. Convert pdfs to text
    3. Chunk large texts
    4. Create and Add docs from chunked text to chroma db
    '''


    # get list of document files
    logger.info("#====================Starting Document Processing=====================")
    file_paths = get_docs_list(pdf_files_path)

     # create chroma vectorstore
    logger.info("#====================Initialising Chroma DB Collection=================")
    db_collection = initialise_collection(db_folder,
                                           collection_name,
                                           embedding_model,
                                           coll_description)
    
    # get embedding function
    logger.info("#====================Creating Embeddings=====================")
    # embedding_function = get_embeding_function(embedding_model)


    # chunk large doc texts and create docs
    for c, path in tqdm(enumerate(file_paths)):
        
        logger.info(f"Processing Document {c+1} of {len(file_paths)}....")
        logger.info(f"file_name: {Path(path).stem}")

        # convert pdf to text
        logger.info("Converting PDF to Text ...")
        full_text = convert_pdf_to_text(path)

        # chunk large doc text
        logger.info("Chunking Doc Text ...")
        list_chunked_text = chunk_text(full_text, chunk_size, overlap)

        # create embedding from chunked text
        logger.info("Creating embeddings from chunked text ...")
        embedded_docs = create_embeddings_from_chunked_text(list_chunked_text, path, embedding_model)

        logger.info("Adding embeddings to vectorstore ...")
        add_doc_embeddings_to_vectorstore(embedded_docs, db_collection)

    logger.info(f"#======Finished processing all {len(file_paths)} Documents======")

    return

    

if __name__ == "__main__":
    process_documents_and_add_to_vectorstore(
        pdf_files_path=INPUT_FOLDER_PATH,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
        db_folder=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        coll_description=COLLECTION_DESCRIPTION,
        chunk_size=MAX_CHARS,
        overlap = OVERLAP
    )
    