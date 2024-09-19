import os
import time
from typing import Annotated
from tavily import TavilyClient
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, Agent, register_function
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor

# Ensure the API keys are set correctly
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("Tavily API key is not set. Please set the TAVILY_API_KEY environment variable.")

# Define configuration list for LLM models
config_list = [
    {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-3.5-turbo-16k", "api_key": os.environ["OPENAI_API_KEY"]},
]

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
    "model": "gpt-3.5-turbo-16k",
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Tavily client setup
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

# NOTE: this ReAct prompt is adapted from Langchain's ReAct agent: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py#L79
ReAct_prompt = """
Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action

... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
"""

def react_prompt_message(sender, recipient, context):
    return ReAct_prompt.format(input=context["question"])

# Setting up code executor.
os.makedirs("coding", exist_ok=True)
# Use docker executor for running code in a container if you have docker installed.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

# Define agents
planner = AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)

scientist = AssistantAgent(
    name="Scientist",
    llm_config=llm_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)

executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

# Custom speaker selection function
def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    messages = groupchat.messages

    if len(messages) <= 1:
        return planner

    if last_speaker is user_proxy:
        if "Approve" in messages[-1]["content"]:
            return engineer
        elif messages[-2]["name"] == "Planner":
            return planner
        elif messages[-2]["name"] == "Scientist":
            return scientist

    elif last_speaker is planner:
        return user_proxy

    elif last_speaker is engineer:
        if "```python" in messages[-1]["content"]:
            return executor
        else:
            return engineer

    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return engineer
        else:
            return scientist

    elif last_speaker is scientist:
        return user_proxy

    else:
        return "random"

groupchat = GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Register the search tool
register_function(
    search_tool,
    caller=planner,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

# Start the interaction with the initial message using retry logic
def initiate_chat_with_retry(agent, manager, message, retries=3, initial_delay=5):
    attempt = 0
    while attempt < retries:
        try:
            agent.initiate_chat(manager, message=message)
            break
        except Exception as e:  # Changed from OpenAIError to Exception to handle all exceptions
            print(f"Attempt {attempt + 1} failed with error: {e}")
            attempt += 1
            if attempt < retries:
                delay = initial_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retry attempts failed.")

# Use disk cache for the chat
with Cache.disk(cache_seed=43) as cache:
    initiate_chat_with_retry(
        user_proxy,
        manager,
        message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.",
        retries=3,
        initial_delay=5
    )
