import os 
import pandas as pd 
from llama_index import ComposableGraph

from src.constants import CSV2TXT_FOLDER 
from src.utils.file_helper import get_filename
from src.AutoCSVPipeline.graph import build_graph_from_indices, save_graph
from src.AutoCSVPipeline.utils import get_embeddings_from_folder, groupby_development, df_row_to_context_file

# TODO: 
def pipeline(
        csv_filepath: str, 
        user_summary_prompt: str = None, 
        index_name: str = None, 
) -> ComposableGraph: 
    # get filename 
    filename = get_filename(csv_filepath) 
    filename = os.path.splitext(filename)[0] # NOTE: remove file extension

    # create stored txt folder 
    stored_txt_folder = f"{CSV2TXT_FOLDER}/{filename}" 
    if not os.path.exists(stored_txt_folder): 
        os.makedirs(stored_txt_folder) 

    # Groupby development 
    df = pd.read_csv(csv_filepath)
    df_of_development = groupby_development(df)

    # Iterate through each development -> convert row to context str 
    # then, save context to txt files
    df_row_to_context_file(df_of_development,
                        to_save_dir=stored_txt_folder) 

    # Indexing by folder & Summarize 
    all_indices, index_summaries = get_embeddings_from_folder(stored_txt_folder,
                                    custom_summary_prompt=user_summary_prompt)  

    # Construct Graph 
    graph = build_graph_from_indices(all_indices,index_summaries)

    # Save Graph 
    graph_name = index_name if index_name else filename
    save_graph(graph, graph_name)

    return graph 