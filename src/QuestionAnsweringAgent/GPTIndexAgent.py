from src.constants import AGENT_VEROBSE
from src.GPTIndexDocument.index_doc import load_index
from src.utils.file_helper import get_filename

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, ConversationalChatAgent, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig


DEFAULT_PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is desgined to do only one job is answering question from the user's document, providing in-depth explanations and discussion on wide range of topics that related to user's document. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. 

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

At the end of your response, you must provide the user 3 following questions about the given topics by this following format: 
----- 
RELEVANT QUESTION: 
1. RELEVANT_QUESTION_1
2. RELEVANT_QUESTION_2
3. RELEVANT_QUESTION_3
----- 


TOOLS:
------

Assistant has access to the following tools:""" 


DEFAULT_FORMAT_INSTRUCTIONS ="""To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```

---------------------------------------
EXAMPLE 1: When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: Hello! How can I assist you today?
```

EXAMPLE 2: When you have to use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: Document Search 
Action Input: "lastest bitcoin price"  
Observation: The database doesn't have information about bitcoin 
```
""" 


DEFAULT_SUFFIX = """Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""


def create_pandas_dataframe_tool(
    filepath: str,
    tool_name: str = None,
    tool_description: str = None, 
    return_direct: bool = True
) -> Tool:

    agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        filepath,
        verbose=True
    )

    df_name = get_filename(filepath)

    if not tool_name:
        tool_name = f"Dataframe of {df_name}"

    if not tool_description:
        tool_description = f"useful when you want to pull out exact number, quotes from the {df_name}. Do not use this tool for exact same input/query."

    tool = Tool(
        name=tool_name,
        description=tool_description,
        return_direct=return_direct, 
        func=agent.run
    )

    return tool


def create_tool_from_index_name(
    index_name: str,
    name: str = None,
    description: str = None,
    top_k: int = 3,
    return_direct: bool = True,
    return_source_documents: bool = False,
) -> LlamaToolkit:
    loaded_index = load_index(index_name=index_name)

    if not name:
        name = f"Vector Index for {index_name} Documents"

    if not description:
        description = f"This is your only tool, useful for when you want to answer queries about the {index_name} documents. DO NOT use this tool for the same input/query. "

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
    index_name: str,
    name: str = None,
    description: str = None,
) -> AgentExecutor:

    toolkit = create_tool_from_index_name(
        index_name=index_name,
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

    return agent_executor


def build_chat_agent_executor(
    index_name: str,
    name: str = None,
    description: str = None,
    additional_tools: list[Tool] = None
) -> AgentExecutor:
    chat_llm = ChatOpenAI(temperature=0.2, max_tokens=None)
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    default_toolkit = create_tool_from_index_name(
        index_name=index_name,
        name=name,
        description=description)
    tools = default_toolkit.get_tools()

    if additional_tools:
        tools.extend(additional_tools)

    agent = ConversationalAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=tools,
        prefix=DEFAULT_PREFIX,
        format_instructions=DEFAULT_FORMAT_INSTRUCTIONS, 
        suffix=DEFAULT_SUFFIX
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=AGENT_VEROBSE
    )
    return agent_executor
