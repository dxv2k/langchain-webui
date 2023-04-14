from langchain import VectorDBQA
from langchain.agents import Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings

from src.IndexDocuments.index_doc import load_index
from src.constants import AGENT_VEROBSE


def create_faiss_tool_from_index_name(
    name: str,  
    description: str,
    index_name: str, 
    top_k: int = 3, 
    return_source_documents: bool = False, 
    chain_type="stuff" 
): 
    completion_llm = OpenAI(temperature=0, max_tokens=-1)

    embeddings = OpenAIEmbeddings()  
    vectorstore = load_index(index_name=index_name, embedding_model=embeddings) 

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


def build_qa_agent_executor(index_name: str) -> AgentExecutor:  
    chat_llm = ChatOpenAI(temperature=0, max_tokens=None)
    
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    tools = [create_faiss_tool_from_index_name( 
                index_name=index_name, 
                name="Document Search" ,
                description="Useful when you want to search for information from documents")
    ]

    agent = ConversationalAgent.from_llm_and_tools( 
        llm=chat_llm, 
        tools=tools
    )

    executor = AgentExecutor.from_agent_and_tools( 
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=AGENT_VEROBSE 
    )

    return executor 

