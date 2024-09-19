import os
import autogen
from autogen import AssistantAgent, ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent, UserProxyAgent
import chromadb

llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],
    "temperature": 0
}

Sales_Rep = AssistantAgent(
    name="Sales_Rep",
    system_message="""You are a sales representative and market expert in understanding the sales pattern in a region and giving advice based on your knowledge.""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

assistant = UserProxyAgent(
    name="assistant",
    human_input_mode="ALWAYS",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    code_execution_config=False,
)

# Defining the task
task = """I am a wholesale seller. My most sold product is a laptop. I belong to the Maharashtra region. My sales are low.
    What are the ideal questions I should be asking to a sales representative?"""

# Initiating chat with the correct assistant
assistant.initiate_chat(Sales_Rep, message=task)
