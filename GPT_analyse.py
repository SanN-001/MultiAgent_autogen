import os
import openai
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

openai.api_key = os.environ.get('OPENAI_API_KEY')

from openai import OpenAI
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent

client = OpenAI(api_key=openai.api_key)

# Corrected file path
file_path = r"C:\Users\sanan\Downloads\PERSONA - Final.csv"

config_list = [
    {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-3.5-turbo-16k", "api_key": os.environ["OPENAI_API_KEY"]},
]

# Read the content of the file to provide context directly
with open(file_path, "r") as file:
    file_content = file.read()

assistant = client.beta.assistants.create(
    name="Python_Developer",
    instructions="You are an expert Python developer. Use the provided data for analysis.",
    model="gpt-3.5-turbo-16k",
    tools=[{"type": "code_interpreter"}]
)

llm_config = {
    "cache_seed": 60,
    "temperature": 1,
    "config_list": config_list,
    "timeout": 600,
    "model": "gpt-3.5-turbo-16k",
    "api_key": os.environ["OPENAI_API_KEY"],
    "assistant_id": assistant.id
}

gpt_assistant = GPTAssistantAgent(
    name="Code Assistant",
    instructions=f"""You are an expert at writing Python code to analyse data.
    Here is the data you need to work with:
    {file_content}
    Reply TERMINATE when the task has been solved.
    """,
    llm_config=llm_config
)

user_proxy = UserProxyAgent(
    name="Jason",
    code_execution_config={
        "work_dir": "coding"
    }
)

task = """Find the orderID that was sold at the highest price."""

user_proxy.initiate_chat(gpt_assistant, message=task)

gpt_assistant.delete_assistant()