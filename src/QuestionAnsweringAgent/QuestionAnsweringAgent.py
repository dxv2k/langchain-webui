from langchain import VectorDBQA
from langchain.agents import Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from src.CustomConversationAgent.CustomConversationAgent import CustomConversationalAgent
from typing import Union

from src.IndexDocuments.index_doc import load_index
from src.constants import AGENT_VEROBSE


CUSTOM_PREFIX = """Assistant is a large language model trained by OpenAI.

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


CUSTOM_FORMAT_INSTRUCTIONS ="""To use a tool, please use the following format:
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


CUSTOM_SUFFIX = """Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""



def create_faiss_tool_from_index_name(
    name: str,  
    description: str,
    index_name: str, 
    top_k: int = 10, 
    return_source_documents: bool = False, 
    chain_type="map_reduce" 
) -> Tool: 
    _validate_chain_type(chain_type) 

    completion_llm = OpenAI(temperature=0, max_tokens=-1)
    if chain_type == "map_reduce": 
        completion_llm = OpenAI(temperature=0)

    embeddings = OpenAIEmbeddings()  
    vectorstore = load_index(index_name=index_name, embedding_model=embeddings) 

    # TODO: replace this with retrievalQA
    search_func = VectorDBQA.from_chain_type(llm=completion_llm,
                                chain_type=chain_type,
                                vectorstore=vectorstore,
                                k=top_k,
                                return_source_documents=return_source_documents
    ).run 

    # def _merge_result_and_source_doc(tool_input: str) -> str: 
    #     query_result: dict = VectorDBQA.from_chain_type(llm=completion_llm,
    #                         chain_type=chain_type,
    #                         vectorstore=vectorstore,
    #                         k=top_k,
    #                         return_source_documents=return_source_documents
    #     ).run(tool_input) 

    #     result: str = query_result.get('result')
    #     source_docs: list[Document] = query_result.get('source_documents')
        
    #     str_source_docs = ""
    #     for doc in source_docs: 
    #         value = doc.metadata + "\n"
    #         str_source_docs += value

    #     merge_result = f"""{result} 
    #     SOURCE DOCUMENTS: {str_source_docs} 
    #     """
    #     return merge_result 


    return Tool( 
        name=name, 
        description=description, 
        func=search_func 
    ) 


def build_qa_agent_executor(index_name: str = None) -> AgentExecutor:  
    chat_llm = ChatOpenAI(temperature=0.2, max_tokens=None)
    
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
    tools = []

    if index_name: 
        tools.append(create_faiss_tool_from_index_name( 
                index_name=index_name, 
                name=f"{index_name} Document Search" ,
                description=f"This is your only tool, useful when you want to search for information from {index_name} documents, don't use this tool for the same input/query.")
        )

    # NOTE: default ConversationAgent
    # agent = ConversationalAgent.from_llm_and_tools( 
    #     llm=chat_llm, 
    #     tools=tools, 
    #     format_instructions=CUSTOM_FORMAT_INSTRUCTIONS
    # )

    # NOTE: CustomConversationAgent -> add more error handling
    agent = CustomConversationalAgent.from_llm_and_tools( 
        llm=chat_llm, 
        tools=tools, 
        prefix=CUSTOM_PREFIX, 
        format_instructions=CUSTOM_FORMAT_INSTRUCTIONS
    )

    executor = AgentExecutor.from_agent_and_tools( 
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=AGENT_VEROBSE 
    )

    return executor 

def _validate_chain_type(chain_type) -> Union[ValueError, None]: 
    allowed_types = ["stuff", "map_reduce", "refine", "map_rerank"]
    if chain_type not in allowed_types:
        raise ValueError(f"Invalid chain_type: {chain_type}. Allowed types are {allowed_types}")
