from typing import Union
import src.GPTIndexDocument.index_doc as gpt_index
from src.constants import FAISS_LOCAL_PATH, SAVE_DIR, GPT_INDEX_LOCAL_PATH
from src.QuestionAnsweringAgent.GPTIndexAgent import build_gpt_index_chat_agent_executor
from src.QuestionAnsweringAgent.QuestionAnsweringAgent import build_qa_agent_executor
import src.IndexDocuments.index_doc as langchain_index
from src.ChatWrapper.ChatWrapper import ChatWrapper
from src.utils.logger import get_logger
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
import gradio as gr
import shutil
import os
os.environ["OPENAI_API_KEY"] = "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"


def prepare_project_dir() -> None:
    if not os.path.exists(FAISS_LOCAL_PATH):
        logger.info(f"created {FAISS_LOCAL_PATH}")
        os.mkdir(FAISS_LOCAL_PATH)

    if not os.path.exists(GPT_INDEX_LOCAL_PATH):
        logger.info(f"created {GPT_INDEX_LOCAL_PATH}")
        os.mkdir(GPT_INDEX_LOCAL_PATH)

    if not os.path.exists(SAVE_DIR):
        logger.info(f"created {SAVE_DIR}")
        os.mkdir(SAVE_DIR)


logger = get_logger()
prepare_project_dir()

UPLOADED_FILES = []
LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH)
GPT_INDEX_LIST_COLLECTIONS = os.listdir(GPT_INDEX_LOCAL_PATH)


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

def load_gpt_index_agent(index_name: str = None) -> AgentExecutor:
    agent_executor = build_gpt_index_chat_agent_executor(index_name=index_name)
    logger.info(f"GPTIndex Agent has access to following tools {agent_executor.tools}")
    logger.info(
        f"Agent used temperature: {agent_executor.agent.llm_chain.llm.temperature}")
    return agent_executor


def load_gpt_index_agent(index_name: str = None) -> AgentExecutor:
    logger.info(
        f"======================Using GPTIndex Agent======================")
    agent_executor = build_gpt_index_chat_agent_executor(index_name=index_name)
    logger.info(f"Agent has access to following tools {agent_executor.tools}")
    logger.info(
        f"Agent used temperature: {agent_executor.agent.llm_chain.llm.temperature}")
    return agent_executor


def get_filename(file_path) -> str:
    return os.path.basename(file_path)


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


def change_gpt_index_agent_handler(index_name: str, chatbot: gr.Chatbot) -> Union[gr.Chatbot, None, None, gr.Slider]:
    logger.info(f"Change GPTIndex Agent to use collection: {index_name}")

    global chat_gpt_index_agent   # NOTE: dirty way to do similar to gr.State()
    chat_gpt_index_agent = None

    agent_executor = load_gpt_index_agent(index_name=index_name)
    chat_gpt_index_agent = ChatWrapper(agent_executor)

    return gr.Chatbot.update(value=[]), None, None, gr.Slider.update(value=agent_executor.agent.llm_chain.llm.temperature)


def refresh_collection_list_handler() -> gr.Dropdown:
    global LIST_COLLECTIONS  # NOTE: dirty way to do similar to gr.State()
    LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH)
    return gr.Dropdown.update(choices=LIST_COLLECTIONS)


def gpt_index_refresh_collection_list_handler() -> gr.Dropdown:
    global GPT_INDEX_LIST_COLLECTIONS  # NOTE: dirty way to do similar to gr.State()
    GPT_INDEX_LIST_COLLECTIONS = os.listdir(GPT_INDEX_LOCAL_PATH)
    return gr.Dropdown.update(choices=GPT_INDEX_LIST_COLLECTIONS)


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


def chat_gpt_index_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_gpt_index_agent
    chatbot, state = chat_gpt_index_agent(message_txt_box, state, agent_state)
    return chatbot, state


def chat_handler(message_txt_box, state, agent_state) -> Union[gr.Chatbot, gr.State]:
    global chat_agent
    chatbot, state = chat_agent(message_txt_box, state, agent_state)
    return chatbot, state


# -------------------------------------------------------------------------------
agent_executor = load_qa_agent("epsom form feed")
chat_agent = ChatWrapper(agent_executor)

gpt_index_agent_executor = load_gpt_index_agent("epsom form feed mini")
chat_gpt_index_agent = ChatWrapper(gpt_index_agent_executor)

# -------------------------------------------------------------------------------


def app() -> gr.Blocks:
    block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

    with block:
        with gr.Tab("Chat GPT_Index"):
            with gr.Row():
                gr.Markdown("<h3><center>GPTIndex + LangChain Demo</center></h3>")

            with gr.Row():
                gpt_index_dropdown_btn = gr.Dropdown(
                    label="Index/Collection to chat with",
                    choices=GPT_INDEX_LIST_COLLECTIONS)

                gpt_refresh_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)

            gpt_temperature_llm_slider = gr.Slider(0, 2, step=0.2, value=0.1, label="Temperature")
            gpt_temperature_llm_slider.change(
                change_temperature_gpt_index_llm_handler,
                inputs=gpt_temperature_llm_slider 
            )

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

        # TODO: to avoid Gradio holding object, you should wrap everything within a function
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

        gpt_index_dropdown_btn.change(change_gpt_index_agent_handler,
                                  inputs=gpt_index_dropdown_btn,
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
                          outputs=gpt_index_dropdown_btn)


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

    logger.info(f"Starting server with config: {args}")

    block = app()

    block.queue(concurrency_count=n_concurrency).launch(
        auth=(username, password),
        debug=debug,
        server_port=server_port,
        show_api=is_show_api
    )

# block = app()
# block.queue(concurrency_count=10).launch(debug=True, server_port=8000)
