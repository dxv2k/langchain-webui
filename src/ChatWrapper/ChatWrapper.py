from typing import Optional, Tuple

from langchain.chains import ConversationChain
from threading import Lock

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
