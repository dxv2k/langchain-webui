from typing import Union
from src.AutoCSVPipeline.agent import build_custom_graph_chat_agent_executor, build_default_graph_chat_agent_executor
from src.AutoCSVPipeline.pipeline import pipeline
from src.AutoCSVPipeline.prompt import SUMMARY_DEVELOPMENT_TEMPLATE_PROMPT
import src.GPTIndexDocument.index_doc as gpt_index
from src.constants import CSV_UPLOADED_FOLDER, FAISS_LOCAL_PATH, KNOWLEDGE_GRAPH_FOLDER, SAVE_DIR, GPT_INDEX_LOCAL_PATH
from src.QuestionAnsweringAgent.GPTIndexAgent import build_chat_agent_executor, build_gpt_index_chat_agent_executor, create_pandas_dataframe_tool
from src.QuestionAnsweringAgent.QuestionAnsweringAgent import build_qa_agent_executor
import src.IndexDocuments.index_doc as langchain_index
from src.ChatWrapper.ChatWrapper import ChatWrapper
from src.utils.prepare_project import prepare_project_dir
from src.utils.logger import get_logger
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
import gradio as gr
import shutil
import os
import dotenv
from os import getenv

dotenv.load_dotenv()
assert getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY not set in .env"


def index_document_from_single_pdf_handler(
        chunk_size: int,
        overlap_chunk: int,
        index_name: str,
        progress=gr.Progress()) -> str:
    global UPLOADED_FILES  # NOTE: dirty way to do similar to gr.State()
    logger.info(
        f"{chunk_size},{overlap_chunk}, {UPLOADED_FILES}, {index_name}")

    progress(0.2, "Indexing Documents....")
    if not index_name:
        filename = get_filename(UPLOADED_FILES[0])
        index_name = os.path.splitext(filename)[0]

    progress(0.5, "Indexing Documents....")
    embeddings = OpenAIEmbeddings()
    faiss_index = langchain_index.single_pdf_indexer(
        filepath=UPLOADED_FILES[0],
        embedding_model=embeddings)

    progress(0.3, "Saving index...")
    langchain_index.save_index(faiss_index, index_name=index_name)
    logger.info(f"Indexing complete & saving {faiss_index}....")
    return "Done!"


def gpt_index_document_from_single_pdf_handler(
        chunk_size: int,
        overlap_chunk: int,
        index_name: str,
        progress=gr.Progress()) -> str:
    global UPLOADED_FILES  # NOTE: dirty way to do similar to gr.State()
    logger.info(
        f"{chunk_size},{overlap_chunk}, {UPLOADED_FILES}, {index_name}")

    progress(0.2, "Verify Documents....")
    if not index_name:
        filename = get_filename(UPLOADED_FILES[0])
        index_name = os.path.splitext(filename)[0]

    progress(0.5, "Analyzing & Indexing Documents....")
    index = gpt_index.single_simple_vector_pdf_indexer(
        filepath=UPLOADED_FILES[0])

    progress(0.3, "Saving index...")
    gpt_index.save_index(index, index_name=index_name)
    logger.info(f"Indexing complete & saving {index}....")
    return "Done!"


def load_simple_chat_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    chat_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    chain = ConversationChain(llm=chat_llm)
    return chain


def load_qa_agent(index_name: str = None) -> AgentExecutor:
    agent_executor = build_qa_agent_executor(index_name=index_name)
    logger.info(f"Agent has access to following tools {agent_executor.tools}")
    logger.info(
        f"Agent used temperature: {agent_executor.agent.llm_chain.llm.temperature}")
    return agent_executor


# NOTE: must remove type annotation when multi-input function
def load_gpt_index_agent(index_name, csv_filepath) -> AgentExecutor:
    logger.info(
        f"======================Using GPTIndex Agent======================")
    additional_tools = []
    if csv_filepath:  
        _path = os.path.join(CSV_UPLOADED_FOLDER,csv_filepath) 
        additional_tools.append(
            create_pandas_dataframe_tool(filepath=_path)) 

    agent_executor = build_chat_agent_executor(
        index_name=index_name, 
        additional_tools=additional_tools
    )
    # agent_executor = build_gpt_index_chat_agent_executor(index_name=index_name)

    logger.info(f"Agent has access to following tools {agent_executor.tools}")
    logger.info(
        f"Agent used temperature: {agent_executor.agent.llm_chain.llm.temperature}")
    return agent_executor


def get_filename(file_path) -> str:
    return os.path.basename(file_path)

def csv_upload_file_handler(files) -> list[str]:
    # global UPLOADED_FILES  # NOTE: dirty way to do similar to gr.State()
    # UPLOADED_FILES = []

    file_paths = [file.name for file in files]

    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(CSV_UPLOADED_FOLDER, filename)

        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    # UPLOADED_FILES = uploads_filepath
    return uploads_filepath

def graph_change_temperature_gpt_index_llm_handler(temperature: float) -> gr.Slider: 
    global chat_graph_agent
    agent_executor = chat_graph_agent.agent
    chat_graph_agent.agent.agent.llm_chain.llm.temperature = temperature
    logger.info(
        f"Change LLM temperature of Graph Agent to {agent_executor.agent.llm_chain.llm.temperature}")


def reset_to_default_prompt_handler() -> gr.Textbox: 
    global USER_SUMMARY_PROMPT
    USER_SUMMARY_PROMPT = SUMMARY_DEVELOPMENT_TEMPLATE_PROMPT 
    return gr.Textbox.update(value=USER_SUMMARY_PROMPT) 


def construct_graph_from_csv_handler(
    index_name, 
    summary_prompt, 
    progress=gr.Progress()
): 
    # global USER_SUMMARY_PROMPT
    global AUTO_CSV_UPLOADED_FILES
    logger.info(
        f"{AUTO_CSV_UPLOADED_FILES}, {index_name}")
    graph = pipeline(
        csv_filepath=AUTO_CSV_UPLOADED_FILES[0], 
        user_summary_prompt=summary_prompt, 
        index_name=index_name 
    ) 

    return "!!! DONE !!!" 


def graph_csv_upload_file_handler(files) -> list[str]: 
    global AUTO_CSV_UPLOADED_FILES  # NOTE: dirty way to do similar to gr.State()
    AUTO_CSV_UPLOADED_FILES = []

    file_paths = [file.name for file in files]

    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(CSV_UPLOADED_FOLDER, filename)

        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    AUTO_CSV_UPLOADED_FILES = uploads_filepath
    print("DEBUG: ",AUTO_CSV_UPLOADED_FILES)
    return uploads_filepath


def upload_file_handler(files) -> list[str]:
    global UPLOADED_FILES  # NOTE: dirty way to do similar to gr.State()
    UPLOADED_FILES = []

    file_paths = [file.name for file in files]

    # create destination directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # loop over all files in the source directory
    uploads_filepath = []
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(SAVE_DIR, filename)

        # copy file from source to destination
        shutil.copy(path, destination_path)
        uploads_filepath.append(destination_path)

    UPLOADED_FILES = uploads_filepath
    return uploads_filepath


def set_openai_api_key(api_key: Union[str,None] = None) -> ConversationChain:
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    chain = load_simple_chat_chain()
    return chain


def change_qa_agent_handler(index_name: str, chatbot: gr.Chatbot) -> Union[gr.Chatbot, None, None, gr.Slider]:
    logger.info(f"Change Agent to use collection: {index_name}")

    global chat_agent  # NOTE: dirty way to do similar to gr.State()
    chat_agent = None

    agent_executor = load_qa_agent(index_name=index_name)
    chat_agent = ChatWrapper(agent_executor)

    return gr.Chatbot.update(value=[]), None, None, gr.Slider.update(value=agent_executor.agent.llm_chain.llm.temperature)


def change_gpt_index_agent_handler(index_name, csv_filepath) -> Union[gr.Chatbot, None, None, gr.Slider]:
    logger.info(f"Change GPTIndex Agent to use collection: {index_name}")

    global chat_gpt_index_agent   # NOTE: dirty way to do similar to gr.State()
    chat_gpt_index_agent = None

    agent_executor = load_gpt_index_agent(index_name=index_name, csv_filepath=csv_filepath)
    chat_gpt_index_agent = ChatWrapper(agent_executor)

    return gr.Chatbot.update(value=[]), None, None, gr.Slider.update(value=agent_executor.agent.llm_chain.llm.temperature)

def change_graph_agent_handler(graph_name) -> Union[gr.Chatbot, None, None, gr.Slider]: 
    logger.info(f"Change Llmama-Index Graph Agent to use collection: {graph_name}")

    global chat_graph_agent   # NOTE: dirty way to do similar to gr.State()
    chat_graph_agent = None

    # agent_executor = build_default_graph_chat_agent_executor(graph_name=graph_name)
    agent_executor = build_custom_graph_chat_agent_executor(graph_name=graph_name)
    chat_graph_agent = ChatWrapper(agent_executor)

    return gr.Chatbot.update(value=[]), None, None, gr.Slider.update(value=agent_executor.agent.llm_chain.llm.temperature)


def refresh_collection_list_handler() -> gr.Dropdown:
    global LIST_COLLECTIONS  # NOTE: dirty way to do similar to gr.State()
    LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH)
    return gr.Dropdown.update(choices=LIST_COLLECTIONS)

def graph_refresh_collection_list_handler() -> gr.Dropdown:
    global KNOWLEDGE_GRAPH_COLLECTIONS  # NOTE: dirty way to do similar to gr.State()
    KNOWLEDGE_GRAPH_COLLECTIONS = os.listdir(KNOWLEDGE_GRAPH_FOLDER)
    return gr.Dropdown.update(choices=KNOWLEDGE_GRAPH_COLLECTIONS)


def gpt_index_refresh_collection_list_handler() -> Union[gr.Dropdown, gr.Dropdown]:
    global GPT_INDEX_LIST_COLLECTIONS  # NOTE: dirty way to do similar to gr.State()
    global CSV_LIST_COLLECTIONS # NOTE: dirty way to do similar to gr.State()

    GPT_INDEX_LIST_COLLECTIONS = os.listdir(GPT_INDEX_LOCAL_PATH)
    CSV_LIST_COLLECTIONS = os.listdir(CSV_UPLOADED_FOLDER)

    return gr.Dropdown.update(choices=GPT_INDEX_LIST_COLLECTIONS), gr.Dropdown.update(choices=CSV_LIST_COLLECTIONS)


def graph_clear_chat_history_handler() -> Union[gr.Chatbot, None, None]:
    global chat_graph_agent  # NOTE: dirty way to do similar to gr.State()
    chat_graph_agent.clear_agent_memory()
    logger.info(f"Clear graph agent memory...")
    return gr.Chatbot.update(value=[]), None, None

def clear_chat_history_handler() -> Union[gr.Chatbot, None, None]:
    global chat_agent  # NOTE: dirty way to do similar to gr.State()
    chat_agent.clear_agent_memory()
    logger.info(f"Clear agent memory...")
    return gr.Chatbot.update(value=[]), None, None


def clear_gpt_index_chat_history_handler() -> Union[gr.Chatbot, None, None]:
    global chat_gpt_index_agent  # NOTE: dirty way to do similar to gr.State()
    chat_gpt_index_agent.clear_agent_memory()
    logger.info(f"Clear agent memory...")
    return gr.Chatbot.update(value=[]), None, None


def change_temperature_llm_handler(temperature: float) -> gr.Slider:
    global chat_agent
    agent_executor = chat_agent.agent
    chat_agent.agent.agent.llm_chain.llm.temperature = temperature
    logger.info(
        f"Change LLM temperature to {agent_executor.agent.llm_chain.llm.temperature}")


def change_temperature_gpt_index_llm_handler(temperature: float) -> gr.Slider:
    global chat_gpt_index_agent
    agent_executor = chat_gpt_index_agent.agent
    agent_executor.agent.llm_chain.llm.temperature = temperature
    logger.info(
        f"Change LLM temperature to {agent_executor.agent.llm_chain.llm.temperature}")


def change_temperature_auto_csv_handler(temperature: float) -> gr.Slider: 
    global chat_graph_agent
    agent_executor = chat_graph_agent.agent
    chat_graph_agent.agent.agent.llm_chain.llm.temperature = temperature
    logger.info(
        f"Change LLM temperature to {agent_executor.agent.llm_chain.llm.temperature}")
    return gr.Slider.update(value=temperature)
    

def chat_gpt_index_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_gpt_index_agent
    chatbot, state = chat_gpt_index_agent(message_txt_box, state, agent_state)
    return chatbot, state


def chat_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_agent
    chatbot, state = chat_agent(message_txt_box, state, agent_state)
    return chatbot, state


def graph_chat_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_graph_agent
    chatbot, state = chat_graph_agent(message_txt_box, state, agent_state)
    return chatbot, state


# -------------------------------------------------------------------------------

def app() -> gr.Blocks:
    block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

    with block:
        with gr.Tab("Chat with Knowledge Graph"):
            with gr.Row():
                graph_index_dropdown_btn = gr.Dropdown(
                    value=KNOWLEDGE_GRAPH_COLLECTIONS[0] if KNOWLEDGE_GRAPH_COLLECTIONS else None, 
                    label="Index/Collection to chat with",
                    choices=KNOWLEDGE_GRAPH_COLLECTIONS)

                graph_refresh_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)

            graph_temperature_llm_slider = gr.Slider(0, 2, step=0.1, value=0.1, label="Temperature")
            graph_temperature_llm_slider.change(
                graph_change_temperature_gpt_index_llm_handler,
                inputs=graph_temperature_llm_slider 
            )

            graph_chatbot = gr.Chatbot()

            with gr.Row():
                    graph_message_txt_box = gr.Textbox(
                        label="What's your question?",
                        placeholder="What's the answer to life, the universe, and everything?",
                        lines=1,
                    ).style(full_width=True)

                    graph_submit_chat_msg_btn = gr.Button(
                        value="Send", variant="primary").style(full_width=False)

                    graph_clear_chat_history_btn = gr.Button(
                        value="Clear chat history (will clear chatbot memory)",
                        variant="stop").style(full_width=False)

            gr.Examples(
                examples=[
                    "Hi! How's it going?",
                    "What should I do tonight?",
                    "Whats 2 + 2?",
                ],
                inputs=graph_message_txt_box,
            )

            gr.HTML("Demo application of a LangChain chain.")
            gr.HTML(
                "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
            )

        with gr.Tab("Construct Knowledge Graph from CSV"):
            gr.Markdown("<h3><center>Construct Knowledge Graph from CSV</center></h3>")
            
            graph_index_name = gr.Textbox(
                label="Collection/Index Name",
                placeholder="What's the name for this index? Eg: Epsom_Reviews_2019",
                lines=1) 

            graph_index_btn = gr.Button(
                value="Index!", variant="primary").style(full_width=False)

            graph_summary_prompt_txt_box = gr.Textbox(
                label="Summary prompt for each development",
                value=USER_SUMMARY_PROMPT,
                lines=8,
            ).style(full_width=True, full_height=True) 

            with gr.Row(): 
                # csv_temperature_llm_slider = gr.Slider(0, 2, step=0.2, value=0.1, label="LLM Temperature")
                # csv_temperature_llm_slider.change(
                #     change_temperature_gpt_index_llm_handler,
                #     inputs=csv_temperature_llm_slider 
                # )

                # update_prompt_btn = gr.Button(value="Update Prompt") 
                # update_prompt_btn.click(update_prompt_handler, 
                #                     inputs=csv_summary_prompt_txt_box)

                default_prompt_btn = gr.Button(value="Reset to default Prompt")
                default_prompt_btn.click(reset_to_default_prompt_handler, 
                    outputs=graph_summary_prompt_txt_box
                )


            graph_file_output = gr.File()
            graph_upload_button = gr.UploadButton(
                "Click to upload *.csv files",
                file_types=[".csv"],
                file_count="multiple"
            )
            graph_upload_button.upload(graph_csv_upload_file_handler, graph_upload_button,
                                graph_file_output, api_name="upload_csv_files")

            graph_index_btn.click(construct_graph_from_csv_handler, 
                                inputs=[graph_index_name,graph_summary_prompt_txt_box], 
                                outputs=graph_index_name
            )


        with gr.Tab("Chat GPT_Index"):
            with gr.Row():
                gr.Markdown("<h3><center>GPTIndex + LangChain Demo</center></h3>")

            csv_file_output = gr.File()
            csv_gpt_upload_button = gr.UploadButton(
                "Click to upload *.csv files",
                file_types=[".csv" ],
                file_count="multiple"
            )
            csv_gpt_upload_button.upload(csv_upload_file_handler, csv_gpt_upload_button,
                                 csv_file_output, api_name="upload_csv_files")

            gr.HTML("Chatbot Config")

            with gr.Row():
                csv_dropdown_btn = gr.Dropdown( 
                    value=CSV_LIST_COLLECTIONS[0] 
                            if CSV_LIST_COLLECTIONS 
                            else None, 
                    label="Dataframe Tool to chat with",
                    choices=CSV_LIST_COLLECTIONS)                

                gpt_index_dropdown_btn = gr.Dropdown(
                    value=GPT_INDEX_LIST_COLLECTIONS[0] \
                            if GPT_INDEX_LIST_COLLECTIONS \
                            else None,  
                    label="Index/Collection (Vector Index Tool) to chat with",
                    choices=GPT_INDEX_LIST_COLLECTIONS)

                gpt_refresh_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)

            gpt_temperature_llm_slider = gr.Slider(0, 2, step=0.2, value=0.1, label="Temperature")
            gpt_temperature_llm_slider.change(
                change_temperature_gpt_index_llm_handler,
                inputs=gpt_temperature_llm_slider 
            )

            gr.HTML("Chat with GPT-Index")
            gpt_index_chatbot = gr.Chatbot()
            with gr.Row():
                gpt_message_txt_box = gr.Textbox(
                    label="What's your question?",
                    placeholder="What's the answer to life, the universe, and everything?",
                    lines=1,
                ).style(full_width=True)

                gpt_submit_chat_msg_btn = gr.Button(
                    value="Send", variant="primary").style(full_width=False)

                gpt_clear_chat_history_btn = gr.Button(
                    value="Clear chat history (will clear chatbot memory)",
                    variant="stop").style(full_width=False)

            gr.Examples(
                examples=[
                    "Hi! How's it going?",
                    "What should I do tonight?",
                    "Whats 2 + 2?",
                ],
                inputs=gpt_message_txt_box,
            )

            gr.HTML("Demo application of a LangChain chain.")
            gr.HTML(
                "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
            )

        css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
        with gr.Tab(css=css, label="GPTIndex Document Indexing"):
            file_output = gr.File()
            gpt_upload_button = gr.UploadButton(
                "Click to upload *.pdf, *.txt files",
                file_types=[".txt", ".pdf"],
                file_count="multiple"
            )
            gpt_upload_button.upload(upload_file_handler, gpt_upload_button,
                                 file_output, api_name="upload_files")
            with gr.Row():
                gpt_chunk_slider = gr.Slider(
                    0, 3500, step=250, value=1000, label="Document Chunk Size")

                gpt_overlap_chunk_slider = gr.Slider(
                    0, 1500, step=20, value=40, label="Overlap Document Chunk Size")

            gpt_index_name = gr.Textbox(
                label="Collection/Index Name",
                placeholder="What's the name for this index? Eg: Document_ABC",
                lines=1)
            gpt_index_doc_btn = gr.Button(
                value="Index!", variant="secondary").style(full_width=False)

            gpt_status_text = gr.Textbox(label="Indexing Status")

            gpt_index_doc_btn.click(gpt_index_document_from_single_pdf_handler,
                                inputs=[gpt_chunk_slider,
                                        gpt_overlap_chunk_slider, gpt_index_name],
                                outputs=gpt_status_text)


        with gr.Tab("Chat (LangChain)"):
            with gr.Row():
                gr.Markdown("<h3><center>LangChain Demo</center></h3>")

            with gr.Row():
                index_dropdown_btn = gr.Dropdown(
                    value=LIST_COLLECTIONS[0]  
                        if LIST_COLLECTIONS \
                        else None,  
                    label="Index/Collection to chat with",
                    choices=LIST_COLLECTIONS)

                refresh_btn = gr.Button(
                    "⟳ Refresh Collections").style(full_width=False)

            temperature_llm_slider = gr.Slider(
                0, 2, step=0.2, value=0.1, label="Temperature")

            temperature_llm_slider.change(
                change_temperature_llm_handler,
                inputs=temperature_llm_slider
            )

            chatbot = gr.Chatbot()
            with gr.Row():
                message_txt_box = gr.Textbox(
                    label="What's your question?",
                    placeholder="What's the answer to life, the universe, and everything?",
                    lines=1,
                ).style(full_width=True)

                submit_chat_msg_btn = gr.Button(
                    value="Send", variant="primary").style(full_width=False)

                clear_chat_history_btn = gr.Button(
                    value="Clear chat history (will clear chatbot memory)",
                    variant="stop").style(full_width=False)

            gr.Examples(
                examples=[
                    "Hi! How's it going?",
                    "What should I do tonight?",
                    "Whats 2 + 2?",
                ],
                inputs=message_txt_box,
            )

            gr.HTML("Demo application of a LangChain chain.")
            gr.HTML(
                "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
            )

        css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
        with gr.Tab(css=css, label="Upload & Index Document (LangChain)"):
            file_output = gr.File()
            upload_button = gr.UploadButton(
                "Click to upload *.pdf, *.txt files",
                file_types=[".txt", ".pdf"],
                file_count="multiple"
            )
            upload_button.upload(upload_file_handler, upload_button,
                                 file_output, api_name="upload_files")
            with gr.Row():
                chunk_slider = gr.Slider(
                    0, 3500, step=250, value=1000, label="Document Chunk Size")

                overlap_chunk_slider = gr.Slider(
                    0, 1500, step=20, value=40, label="Overlap Document Chunk Size")

            index_name = gr.Textbox(
                label="Collection/Index Name",
                placeholder="What's the name for this index? Eg: Document_ABC",
                lines=1)
            index_doc_btn = gr.Button(
                value="Index!", variant="secondary").style(full_width=False)

            status_text = gr.Textbox(label="Indexing Status")

            index_doc_btn.click(index_document_from_single_pdf_handler,
                                inputs=[chunk_slider,
                                        overlap_chunk_slider, index_name],
                                outputs=status_text)

        # NOTE: llama-index Graph Chat Agent
        graph_state = gr.State()
        graph_agent_state = gr.State()

        graph_index_dropdown_btn.change(change_graph_agent_handler,
                                  inputs=graph_index_dropdown_btn,
                                  outputs=[graph_chatbot, graph_state, graph_agent_state, graph_temperature_llm_slider])

        graph_clear_chat_history_btn.click(
            graph_clear_chat_history_handler,
            outputs=[graph_chatbot, graph_state, graph_agent_state]
        )

        graph_refresh_btn.click(fn=graph_refresh_collection_list_handler,
                          outputs=graph_index_dropdown_btn)

        # NOTE: to avoid Gradio holding object, you should wrap everything within a function
        graph_submit_chat_msg_btn.click(fn=graph_chat_handler,
                                  inputs=[graph_message_txt_box, graph_state, graph_agent_state],
                                  outputs=[graph_chatbot, graph_state])

        # NOTE: to avoid Gradio holding object, you should wrap everything within a function
        graph_message_txt_box.submit(fn=graph_chat_handler,
                               inputs=[graph_message_txt_box, graph_state, graph_agent_state],
                               outputs=[graph_chatbot, graph_state],
                               api_name="chats")
 
        # LangChain Chat Agent
        state = gr.State()
        agent_state = gr.State()

        index_dropdown_btn.change(change_qa_agent_handler,
                                  inputs=index_dropdown_btn,
                                  outputs=[chatbot, state, agent_state, temperature_llm_slider])

        clear_chat_history_btn.click(
            clear_chat_history_handler,
            outputs=[chatbot, state, agent_state]
        )

        refresh_btn.click(fn=refresh_collection_list_handler,
                          outputs=index_dropdown_btn)

        # NOTE: to avoid Gradio holding object, you should wrap everything within a function
        submit_chat_msg_btn.click(fn=chat_handler,
                                  inputs=[message_txt_box, state, agent_state],
                                  outputs=[chatbot, state])

        message_txt_box.submit(fn=chat_handler,
                               inputs=[message_txt_box, state, agent_state],
                               outputs=[chatbot, state],
                               api_name="chats")

        # NOTE: GPT Index
        gpt_state = gr.State()
        gpt_agent_state = gr.State()

        # NOTE: Multi inputs function must remove type annotation 
        csv_dropdown_btn.change(change_gpt_index_agent_handler, 
                            inputs=[gpt_index_dropdown_btn,csv_dropdown_btn],
                            outputs=[gpt_index_chatbot, gpt_state, gpt_agent_state, gpt_temperature_llm_slider])


        gpt_index_dropdown_btn.change(change_gpt_index_agent_handler,
                                  inputs=[gpt_index_dropdown_btn, csv_dropdown_btn],
                                  outputs=[gpt_index_chatbot, gpt_state, gpt_agent_state, gpt_temperature_llm_slider])


        gpt_submit_chat_msg_btn.click(chat_gpt_index_handler,
                                      inputs=[gpt_message_txt_box,
                                              gpt_state, gpt_agent_state],
                                      outputs=[gpt_index_chatbot, gpt_state])

        gpt_message_txt_box.submit(chat_gpt_index_handler,
                                   inputs=[gpt_message_txt_box,
                                           gpt_state, gpt_agent_state],
                                   outputs=[gpt_index_chatbot, gpt_state],
                                   api_name="chats_gpt_index")

        gpt_refresh_btn.click(fn=gpt_index_refresh_collection_list_handler,
                          outputs=[gpt_index_dropdown_btn, csv_dropdown_btn])


        gpt_clear_chat_history_btn.click(
            clear_gpt_index_chat_history_handler,
            outputs=[gpt_index_chatbot, gpt_state, gpt_agent_state]
        )

    return block


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Launch block queue with authentication and server details")

    parser.add_argument("--username", dest="username",
                        default="admin", help="Authentication username")
    parser.add_argument("--password", dest="password",
                        default="1234@abcezIJK1", help="Authentication password")
    parser.add_argument("--concurrency", dest="concurrency", default=1,
                        type=int, help="Number of concurrent blocks to process")
    parser.add_argument("--debug", dest="debug",
                        action="store_true", help="Enable debug mode")
    parser.add_argument("--port", dest="port", default=8000,
                        type=int, help="Server port")
    parser.add_argument("--show-api", dest="show_api",
                        action="store_true", help="Show API details")

    args = parser.parse_args()

    # Usage:
    # python script.py --username admin --password 1234@abcezIJK1 --concurrency 10 --debug --port 8000 --show-api
    # or
    # python script.py -u admin -p 1234@abcezIJK1 -c 10 -d -o 8000 -s

    n_concurrency = args.concurrency
    username = args.username
    password = args.password
    debug = args.debug
    server_port = args.port
    is_show_api = args.show_api

    logger = get_logger()
    prepare_project_dir(logger=logger)

    logger.info(f"Starting server with config: {args}")

    # Declared global variable scope
    UPLOADED_FILES = []


    LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH)
    agent_executor = load_qa_agent(LIST_COLLECTIONS[0])
    chat_agent = ChatWrapper(agent_executor)

    GPT_INDEX_LIST_COLLECTIONS = os.listdir(GPT_INDEX_LOCAL_PATH)
    CSV_LIST_COLLECTIONS = os.listdir(CSV_UPLOADED_FOLDER) 
    gpt_index_agent_executor = load_gpt_index_agent(GPT_INDEX_LIST_COLLECTIONS[0], 
                                                    CSV_LIST_COLLECTIONS[0])
    chat_gpt_index_agent = ChatWrapper(gpt_index_agent_executor)

    AUTO_CSV_UPLOADED_FILES = []
    KNOWLEDGE_GRAPH_COLLECTIONS = os.listdir(KNOWLEDGE_GRAPH_FOLDER)
    USER_SUMMARY_PROMPT = SUMMARY_DEVELOPMENT_TEMPLATE_PROMPT 
    chat_graph_agent = None
    if KNOWLEDGE_GRAPH_COLLECTIONS: 
        # graph_executor = build_default_graph_chat_agent_executor(KNOWLEDGE_GRAPH_COLLECTIONS[0])  
        graph_executor = build_custom_graph_chat_agent_executor(KNOWLEDGE_GRAPH_COLLECTIONS[0])  
        chat_graph_agent = ChatWrapper(graph_executor) 

    block = app()
    block.queue(concurrency_count=n_concurrency).launch(
        auth=(username, password),
        debug=debug,
        server_port=server_port,
        show_api=is_show_api
    )

# NOTE: for development
# block = app()
# block.queue(concurrency_count=10).launch(debug=True, server_port=8000)
