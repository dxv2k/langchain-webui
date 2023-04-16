import os
os.environ["OPENAI_API_KEY"] = "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"

import shutil
import gradio as gr


from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.embeddings.openai import OpenAIEmbeddings


from src.constants import FAISS_LOCAL_PATH, SAVE_DIR
from src.ChatWrapper.ChatWrapper import ChatWrapper
from src.QuestionAnsweringAgent.QuestionAnsweringAgent import build_qa_agent_executor
from src.IndexDocuments.index_doc import save_index, single_pdf_indexer

def prepare_project_dir() -> None: 
    if not os.path.exists(FAISS_LOCAL_PATH): 
        print(f"created {FAISS_LOCAL_PATH}")
        os.mkdir(FAISS_LOCAL_PATH)

    if not os.path.exists(SAVE_DIR): 
        print(f"created {SAVE_DIR}")
        os.mkdir(SAVE_DIR)


prepare_project_dir() 

UPLOADED_FILES = []
LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH) 

def index_document_from_single_pdf_handler(
        chunk_size: int,
        overlap_chunk: int,
        index_name: str, 
        progress= gr.Progress()) -> str:
    global UPLOADED_FILES # NOTE: dirty way to do similar to gr.State()
    print(f"{chunk_size},{overlap_chunk}, {UPLOADED_FILES}, {index_name}")

    progress(0.2, "Indexing Documents....")
    if not index_name: 
        filename = get_filename(UPLOADED_FILES[0])
        index_name = os.path.splitext(filename)[0]
        
    progress(0.5, "Indexing Documents....")
    embeddings = OpenAIEmbeddings()
    faiss_index = single_pdf_indexer(
        filepath=UPLOADED_FILES[0],
        embedding_model=embeddings)

    progress(0.3, "Saving index...")
    save_index(faiss_index, index_name=index_name)

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
    return agent_executor


def get_filename(file_path) -> str:
    return os.path.basename(file_path)


def upload_file_handler(files) -> list[str]:
    global UPLOADED_FILES # NOTE: dirty way to do similar to gr.State()
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


def set_openai_api_key(api_key: str | None = None) -> ConversationChain:
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    chain = load_simple_chat_chain()
    return chain


def change_qa_agent_handler(index_name: str, chatbot: gr.Chatbot) -> gr.Chatbot: 
    print(f"Change Agent to use collection: {index_name}")
    print(f"Change Agent to use collection: {os.path.join(FAISS_LOCAL_PATH,index_name)}")

    global chat_agent # NOTE: dirty way to do similar to gr.State() 
    chat_agent = None

    agent_executor = load_qa_agent(index_name=index_name)
    chat_agent = ChatWrapper(agent_executor)
    
    #TODO: clean up chat history from the UI too 
    return gr.Chatbot.update(value=[]), None, None 


def refresh_collection_list_handler() -> gr.Dropdown: 
    global LIST_COLLECTIONS # NOTE: dirty way to do similar to gr.State()
    LIST_COLLECTIONS = os.listdir(FAISS_LOCAL_PATH) 
    return gr.Dropdown.update(choices=LIST_COLLECTIONS)

def clear_chat_history_handler(): 
    global chat_agent # NOTE: dirty way to do similar to gr.State()
    chat_agent.clear_agent_memory()
    return gr.Chatbot.update(value=[]), None, None 




agent_executor = load_qa_agent()
chat_agent = ChatWrapper(agent_executor)

def app() -> gr.Blocks: 

    # chain = set_openai_api_key()

    global chat_agent # NOTE: dirty way to do similar to gr.State()

    block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

    with block:
        with gr.Tab("Chat"):
            with gr.Row():
                gr.Markdown("<h3><center>LangChain Demo</center></h3>")

            with gr.Row(): 
                index_dropdown_btn = gr.Dropdown(
                    label="Index/Collection to chat with", 
                    choices=LIST_COLLECTIONS)            
                
                refresh_btn = gr.Button("⟳ Refresh Collections").style(full_width=False)

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
                    value="Clear chat history (will clear chatbot memory)", variant="stop").style(full_width=False)

                

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
        with gr.Tab(css=css, label="Upload & Index Document"):
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
                    0, 3500, step=250, value=1500, label="Document Chunk Size")

                overlap_chunk_slider = gr.Slider(
                    0, 1500, step=20, value=200, label="Overlap Document Chunk Size")

            index_name = gr.Textbox(
                label="Collection/Index Name",
                placeholder="What's the name for this index? Eg: Document_ABC",
                lines=1)
            index_doc_btn = gr.Button(
                value="Index!", variant="secondary").style(full_width=False)

            status_text = gr.Textbox(label="Indexing Status")

            index_doc_btn.click(index_document_from_single_pdf_handler,
                            inputs=[chunk_slider, overlap_chunk_slider, index_name], 
                            outputs=status_text)


        state = gr.State()
        agent_state = gr.State()

        index_dropdown_btn.change(change_qa_agent_handler,
                                inputs=index_dropdown_btn, 
                                outputs=[chatbot,state,agent_state])
        clear_chat_history_btn.click(
                    clear_chat_history_handler, 
                    outputs=[chatbot,state, agent_state]
        )

        refresh_btn.click(fn=refresh_collection_list_handler, 
                            outputs=index_dropdown_btn)

        submit_chat_msg_btn.click(chat_agent,
                                inputs=[message_txt_box, state, agent_state],
                                outputs=[chatbot, state])

        message_txt_box.submit(chat_agent,
                            inputs=[message_txt_box, state, agent_state],
                            outputs=[chatbot, state],
                            api_name="chats")

    return block

if __name__ == "__main__": 
    
    import argparse

    parser = argparse.ArgumentParser(description="Launch block queue with authentication and server details")

    parser.add_argument("--username", dest="username", default="admin", help="Authentication username")
    parser.add_argument("--password", dest="password", default="1234@abcezIJK1", help="Authentication password")
    parser.add_argument("--concurrency", dest="concurrency", default=10, type=int, help="Number of concurrent blocks to process")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", dest="port", default=8000, type=int, help="Server port")
    parser.add_argument("--show-api", dest="show_api", action="store_true", help="Show API details")

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

    print(args)

    block = app() 
    
    block.queue(concurrency_count=n_concurrency).launch(
        auth=(username,password), 
        debug=debug, 
        server_port=server_port, 
        show_api=is_show_api
    )

# block = app()
# block.queue(concurrency_count=10).launch(debug=True, server_port=8000)