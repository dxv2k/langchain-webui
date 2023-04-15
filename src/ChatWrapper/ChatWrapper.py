from typing import Optional, Tuple

from threading import Lock
from langchain.chains import ConversationChain


from src.CustomConversationAgent.CustomConversationAgent import CustomConversationalAgent

class ChatWrapper:

    def __init__(self, 
                 agent: CustomConversationalAgent = None):
        self.lock = Lock()
        self.agent = agent    

    def __call__(
            self, 
            inp: str, 
            history: Optional[Tuple[str, str]], 
            agent: Optional[CustomConversationalAgent]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []

            if agent: self.agent = agent

            # If chain is None, that is because no API key was provided.
            if self.agent is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history

            # Run chain and append input.
            output = self.agent.run(input=inp)
            history.append((inp, output))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history
