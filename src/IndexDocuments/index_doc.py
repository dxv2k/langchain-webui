from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, VectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import os

from src.constants import FAISS_LOCAL_PATH

# os.environ["OPENAI_API_KEY"] = "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"


def single_ppt_indexer( 
        filepath: str,
        embedding_model: OpenAIEmbeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 250
) -> VectorStore:
    raise NotImplementedError 

def single_text_indexer( 
        filepath: str,
        embedding_model: OpenAIEmbeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 250
) -> VectorStore:
    raise NotImplementedError 


def single_pdf_indexer(
        filepath: str,
        embedding_model: OpenAIEmbeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 250
) -> VectorStore:

    loader = PyPDFLoader(filepath)
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pages = loader.load_and_split(text_splitter=text_splitter)
    faiss_index = FAISS.from_documents(pages, embedding_model)
    return faiss_index


def save_index(index: VectorStore, index_name: str) -> None:
    saved_path = os.path.join(FAISS_LOCAL_PATH, index_name) 

    if not os.path.exists(saved_path): 
        os.mkdir(saved_path)

    FAISS.save_local(
        index, 
        folder_path=saved_path, 
    )


def load_index(index_name: str, embedding_model: OpenAIEmbeddings) -> VectorStore: 
    loaded_path = os.path.join(FAISS_LOCAL_PATH,index_name)
    if not os.path.exists(loaded_path): 
        raise ValueError(f"{loaded_path} not exists")

    faiss_index = FAISS.load_local(loaded_path, embedding_model) 
    return faiss_index 
