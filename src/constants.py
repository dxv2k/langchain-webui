FAISS_LOCAL_PATH: str = "./faiss"

GPT_INDEX_LOCAL_PATH: str = "./GPTIndexEmbeddings"

SAVE_DIR: str = "./uploads/"

CSV_UPLOADED_FOLDER: str = "./uploaded_csv"

AGENT_VEROBSE: bool = True

CSV2TXT_FOLDER = "./csv2txt"

KNOWLEDGE_GRAPH_FOLDER: str = "./knowledge_graph" 

GPT_INDEX_QUERY_CONFIG = query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 5,
            # "include_summary": True
        },
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        }
    },
]
