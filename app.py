import os
import shutil
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock


SAVE_DIR = "./uploads/"

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


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

def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else: 
        os.environ["OPENAI_API_KEY"] = "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"  

    chain = load_chain()
    return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
            self, 
            api_key: str, 
            inp: str, 
            history: Optional[Tuple[str, str]], 
            chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
    with gr.Tab(css=css,label="Upload Documents") as demo:
        file_output = gr.File()
        upload_button = gr.UploadButton(
                    "Click to *.pdf, *.txt files", 
                    file_types=[".txt", ".pdf"], 
                    file_count="multiple", 
                    save_to=SAVE_DIR)
        upload_button.upload(upload_file_handler, upload_button, file_output)

    with gr.Tab("Chat"): 
        with gr.Row():
            gr.Markdown("<h3><center>LangChain Demo</center></h3>")

            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...)",
                show_label=False,
                lines=1,
                type="password",
            )

        chatbot = gr.Chatbot()
        with gr.Row():
            message = gr.Textbox(
                label="What's your question?",
                placeholder="What's the answer to life, the universe, and everything?",
                lines=1,
            )
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

        gr.Examples(
            examples=[
                "Hi! How's it going?",
                "What should I do tonight?",
                "Whats 2 + 2?",
            ],
            inputs=message,
        )

        gr.HTML("Demo application of a LangChain chain.")

        gr.HTML(
            "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
        )

        state = gr.State()
        agent_state = gr.State()

        submit.click(chat, 
                    inputs=[openai_api_key_textbox, message, state, agent_state], 
                    outputs=[chatbot, state]
        )

        message.submit(chat, 
                    inputs=[openai_api_key_textbox, message, state, agent_state], 
                    outputs=[chatbot, state]
        )

        openai_api_key_textbox.change(
            set_openai_api_key,
            inputs=[openai_api_key_textbox],
            outputs=[agent_state],
        )

block.launch(debug=True, server_port=8000)
