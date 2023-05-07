FAISS_LOCAL_PATH: str = "./faiss"

GPT_INDEX_LOCAL_PATH: str = "./GPTIndexEmbeddings"

GRAPH_EMBEDDINGS_LOCAL_PATH: str = "./LlamaGraphEmbeddings"

SAVE_DIR: str = "./uploads/"

CSV_UPLOADED_FOLDER: str = "./uploaded_csv"

AGENT_VEROBSE: bool = True

CSV2TXT_FOLDER = "./csv2txt"

KNOWLEDGE_GRAPH_FOLDER: str = "./knowledge_graph" 

GRAPH_QUERY_CONFIG = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 10,
            "response_mode": "tree_summarize"
        }
    },
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 10,
            "response_mode": "tree_summarize"
        }
    },
] 