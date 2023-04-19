from src.constants import AGENT_VEROBSE
from src.IndexDocuments.index_doc import load_index

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory


from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig



def create_tool_from_index_name(
    name: str,
    description: str,
    index_name: str,
    top_k: int = 3,
    return_direct: bool = True,
    return_source_documents: bool = False,
) -> LlamaToolkit:
    loaded_index = load_index(index_name=index_name)
    tool_config = [IndexToolConfig(
        index=loaded_index,
        name=name,
        description=description,
        index_query_kwargs={"similarity_top_k": top_k},
        tool_kwargs={
            "return_direct": return_direct,
            "return_sources": return_source_documents},
    )]

    toolkit = LlamaToolkit(
        index_configs=tool_config,
    )
    return toolkit


# NOTE: to maintain similar name convention
def build_gpt_index_chat_agent_executor(
    index_name: str
) -> AgentExecutor:

    toolkit = create_tool_from_index_name(index_name=index_name)
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0.2)
    agent_executor = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=AGENT_VEROBSE
    )
    return agent_executor
