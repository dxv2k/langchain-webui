import os
import pandas as pd 
from tqdm import tqdm
from llama_index import SimpleDirectoryReader, GPTTreeIndex, GPTVectorStoreIndex, ComposableGraph, GPTListIndex
from src.constants import CSV2TXT_FOLDER, GPT_INDEX_LOCAL_PATH, KNOWLEDGE_GRAPH_FOLDER, GPT_INDEX_LOCAL_PATH 
from src.AutoCSVPipeline.prompt import get_summary_prompt 


def groupby_development(df: pd.DataFrame) -> dict[str | int,pd.DataFrame]: 
    df_of_development: dict = {} 
    development_id = df['development_id'].unique()
    for id in development_id: 
        data = df[ 
            df['development_id'] == id 
        ]
        df_of_development[id] = data
    return df_of_development 



def df_row_to_context_file(
        df_of_development: dict[str | int,pd.DataFrame], 
        to_save_dir: str
): 
    """
        Convert Dataframe Row to text and save to file by {development_id}_{reviewer_id}.txt 
        param: 
            df_of_development: 
            to_save_dir 
        return: 
    """
    dict_context_by_development: dict[str,list[str]] = {}
    for id in tqdm(list(df_of_development.keys())):      
        _df = df_of_development[id]
        for idx, row in _df.iterrows(): 
            reviewer_id = row['id']
            review_date = row['review_date']
            development = row['development']
            development_id = row['development_id']
            rating = row['rating']
            rating_facilities = row['rating_facilities']
            rating_design = row['rating_design']
            rating_location = row['rating_location']
            rating_value = row['rating_value']
            rating_management = row['rating_management']
            tenant_recommends = row['tenant_recommends']
            title = row['title']
            review_content = row['review_content']
            landlord_comments = row['landlord_comments']
            one_thing = row['one_thing']
            best_feature = row['best_feature']

            context_str_from_row = f"""This is another resident’s review of development: {development} (id: {development_id}) written on {review_date} with reviewer ID: {reviewer_id} 
This reviewer scored the development as follows: 
Overall: {rating}/5 
Facilities: {rating_facilities}/5  
Design: {rating_design}/5 
Location: {rating_location}/5 
Value: {rating_value}/5 
Management: {rating_management}/5  
Would the tenant recommend this landlord: {tenant_recommends} 

------
Reviewer's comment: 
Review Title: {title} 
Comment: 
{review_content} 
------

Review of the Landlord: {landlord_comments} 

One thing which surprised the reviewer: {one_thing} 

The best feature of the development: {best_feature} 
"""
            dict_context_by_development[
                f"{development_id}_{reviewer_id}"] = context_str_from_row 

            # Create directory if it doesn't exist
            directory = f"{to_save_dir}/{development_id}/"
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Write context_template to file
            filename = f"{development_id}_{reviewer_id}.txt"
            with open(f"{directory}/{filename}", "w+", encoding="utf-8") as f:
                f.write(context_str_from_row)

    return dict_context_by_development

def get_embeddings_from_folder(
        folderpath: str, 
        custom_summary_prompt: str = None
) -> tuple[list[GPTVectorStoreIndex], list[str]]: 
    list_developments = os.listdir(folderpath)

    all_indices: list[GPTVectorStoreIndex] = []
    index_summaries: list[str] = []
    for development_id in tqdm(list_developments): 
        documents = SimpleDirectoryReader(f"./{folderpath}/{development_id}/").load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
        index.index_struct.index_id = f"{development_id}_development" 
        
        summary_prompt = get_summary_prompt(development_id=development_id, 
            custom_body_prompt=custom_summary_prompt)

        summary = index.query(summary_prompt) 
        summary = summary.response # NOTE: to get string only

        all_indices.append(index)
        index_summaries.append(summary)
        index.save_to_disk(f"./{GPT_INDEX_LOCAL_PATH}/{development_id}.json")

    return all_indices, index_summaries

