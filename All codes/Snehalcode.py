import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os
import chromadb

llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],
    "temperature": 0
}

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are given persona of sales repo of 6 employees , give specific info.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],
    },
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": ["PERSONA.csv"
                      #, os.path.join(os.path.abspath(""), "..", "website", "docs"),
                      ],
        "chunk_token_size": 2000
        ,
        "model": "gpt-3.5-turbo",
        "client": chromadb.PersistentClient(path="./h"),
        "get_or_create": True

    },
    code_execution_config=False,
)
assistant.reset()
qa_problem = " Employee Davolio	Nancy, total quantity sold to all customers"
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=qa_problem)
