import os
import shutil
from typing import Optional, Tuple

import gradio as gr
from threading import Lock


from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader

os.environ["OPENAI_API_KEY"] = "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"  
SAVE_DIR = "./uploads/"

def load_simple_chat_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    chat_llm = ChatOpenAI(
        temperature=0, 
        model_name = "gpt-3.5-turbo"
    ) 
    chain = ConversationChain(llm=chat_llm)
    return chain

def extract_embedding_from_docs(docname: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> OpenAIEmbeddings:  
    # NOTE: pseudo func     

    def _get_file_from_docname(docname: str) -> str: 
        pass 

    embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")

    doc_path = _get_file_from_docname(docname)
    
    if not doc_path: 
        raise ValueError(f"document {docname} is not exists in the storage.") 

    # TODO: process like normal
    loader = UnstructuredFileLoader(doc_path)

    doc_embeds = None
    return doc_embeds


def get_filename(file_path):
    return os.path.basename(file_path)

def upload_file_handler(files):
    file_paths = [file.name for file in files]

    # create destination directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # loop over all files in the source directory
    for path in file_paths:
        filename = get_filename(path)
        destination_path = os.path.join(SAVE_DIR, filename)

        # copy file from source to destination
        shutil.copy(path, destination_path)

    return file_paths

def set_openai_api_key(api_key: str | None = None) -> ConversationChain:
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    chain = load_simple_chat_chain()
    return chain

class ChatWrapper:

    def __init__(self, chain: ConversationChain = None):
        self.lock = Lock()
        self.chain = chain    

    def __call__(
            self, 
            inp: str, 
            history: Optional[Tuple[str, str]], 
            chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []

            if chain: self.chain = chain

            # If chain is None, that is because no API key was provided.
            if self.chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history

            # Run chain and append input.
            output = self.chain.run(input=inp)
            history.append((inp, output))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chain = set_openai_api_key() 
chat = ChatWrapper(chain=chain)

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")
file_options = os.listdir(SAVE_DIR)

with block:
    with gr.Tab("Chat"): 
        with gr.Row():
            gr.Markdown("<h3><center>LangChain Demo</center></h3>")

        chatbot = gr.Chatbot()
        with gr.Row():
            message_txt_box = gr.Textbox(
                label="What's your question?",
                placeholder="What's the answer to life, the universe, and everything?",
                lines=1,
            )
            submit_chat_msg_btn = gr.Button(value="Send", variant="secondary").style(full_width=False)

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
    with gr.Tab(css=css,label="Upload & Index Document"):
        file_output = gr.File()
        upload_button = gr.UploadButton(
                    "Click to upload *.pdf, *.txt files", 
                    file_types=[".txt", ".pdf"], 
                    file_count="multiple")
        upload_button.upload(upload_file_handler, upload_button, file_output, api_name="upload_files")
        with gr.Row():
            chunk_slider = gr.Slider(0, 3000, step=250, label="Document Chunk Size")
            chunk_state = gr.State(value=300)

            overlap_chunk_slider = gr.Slider(0, 1250, step=20, label="Overlap Document Chunk Size")
            overlap_chunk_state = gr.State(value=20)


        index_name = gr.Textbox(
            label="Collection/Index Name",
            placeholder="What's the name for this index? Eg: Document_ABC",
            lines=1,
        )
        index_doc_btn = gr.Button(value="Index!", variant="secondary").style(full_width=False)
        # TODO: execute index function here


    state = gr.State()
    agent_state = gr.State()
 

    submit_chat_msg_btn.click(chat, 
                inputs=[message_txt_box, state, agent_state], 
                outputs=[chatbot, state]
    )

    message_txt_box.submit(chat, 
                inputs=[message_txt_box, state, agent_state], 
                outputs=[chatbot, state], 
                api_name="chats"
    )

block.launch(debug=True, server_port=8000, show_api=True)
