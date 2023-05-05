import os 
from src.constants import CSV_UPLOADED_FOLDER, FAISS_LOCAL_PATH, KNOWLEDGE_GRAPH_FOLDER, SAVE_DIR, GPT_INDEX_LOCAL_PATH, CSV2TXT_FOLDER


def prepare_project_dir(logger: any) -> None:
    if not os.path.exists(FAISS_LOCAL_PATH):
        logger.info(f"created {FAISS_LOCAL_PATH}")
        os.mkdir(FAISS_LOCAL_PATH)

    if not os.path.exists(GPT_INDEX_LOCAL_PATH):
        logger.info(f"created {GPT_INDEX_LOCAL_PATH}")
        os.mkdir(GPT_INDEX_LOCAL_PATH)

    if not os.path.exists(SAVE_DIR):
        logger.info(f"created {SAVE_DIR}")
        os.mkdir(SAVE_DIR)
    
    if not os.path.exists(CSV_UPLOADED_FOLDER):
        logger.info(f"created {CSV_UPLOADED_FOLDER}")
        os.mkdir(CSV_UPLOADED_FOLDER)

    if not os.path.exists(CSV2TXT_FOLDER):
        logger.info(f"created {CSV2TXT_FOLDER}")
        os.mkdir(CSV2TXT_FOLDER)

    if not os.path.exists(KNOWLEDGE_GRAPH_FOLDER):
        logger.info(f"created {KNOWLEDGE_GRAPH_FOLDER}")
        os.mkdir(KNOWLEDGE_GRAPH_FOLDER)

