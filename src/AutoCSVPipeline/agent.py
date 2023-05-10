from src.AutoCSVPipeline.agent_prompt import DEFAULT_FORMAT_INSTRUCTIONS, DEFAULT_PREFIX, DEFAULT_SUFFIX
from src.AutoCSVPipeline.graph import load_graph
from src.constants import AGENT_VEROBSE, GRAPH_QUERY_CONFIG

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ConversationalChatAgent, ConversationalAgent
from llama_index.langchain_helpers.agents import create_llama_chat_agent, GraphToolConfig, LlamaGraphTool, LlamaToolkit


def create_tool_from_graph_name(
    graph_name: str,
    name: str = None,
    description: str = None,
    return_direct: bool = True,
    return_source_documents: bool = False,
) -> LlamaToolkit:
    graph = load_graph(graph_name=graph_name)

    if not name:
        name = f"Vector Index for {graph_name} Documents"

    if not description:
        description = f"This is your only tool, useful for when you want to answer queries about the {graph_name} documents. DO NOT use this tool for the same input/query. "

    tool_config = GraphToolConfig(
        graph=graph,
        name=name,
        description=description,
        query_configs=GRAPH_QUERY_CONFIG, 
        tool_kwargs={
            "return_direct": return_direct,
            "return_sources": return_source_documents},
    )

    graph_tool = LlamaGraphTool.from_tool_config(
        tool_config=tool_config,
    )
    
    toolkit = LlamaToolkit(
        graph_configs=[graph_tool]
    )

    return toolkit


def build_default_graph_chat_agent_executor(
    graph_name: str,
    name: str = None,
    description: str = None,
) -> AgentExecutor:

    toolkit = create_tool_from_graph_name(
        graph_name=graph_name,
        name=name,
        description=description)

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0.2)
    agent_executor = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=AGENT_VEROBSE
    )
    
    # NOTE: temporary fix because can't change this flag with GraphTool
    agent_executor.tools[0].return_direct = True

    return agent_executor


def build_custom_graph_chat_agent_executor(
    graph_name: str,
    name: str = None,
    description: str = None,
) -> AgentExecutor:

    toolkit = create_tool_from_graph_name(
        graph_name=graph_name,
        name=name,
        description=description)
    tools = toolkit.get_tools()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_llm = ChatOpenAI(temperature=0.2)
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=tools,
        system_message=DEFAULT_PREFIX,
        # suffix=DEFAULT_SUFFIX
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=AGENT_VEROBSE
    )
    
    # NOTE: temporary fix because can't change this flag with GraphTool
    agent_executor.tools[0].return_direct = True

    return agent_executor