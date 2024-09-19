import os
from typing import Annotated
import autogen
from tavily import TavilyClient
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.agentchat.contrib.capabilities import teachability

# Ensure the API keys are set correctly
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Define configuration list for LLM models
config_list = [
    {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
]

# Initialize Tavily client (if needed for any web search in the future)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define the search tool function (optional, if needed for any search tasks)
def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

# Setting up code executor
os.makedirs("coding", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

# Initialize UserProxyAgent
user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

# Initialize AssistantAgent with teachability capability
assistant = AssistantAgent(
    name="Assistant",
    system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

# Register the search tool with the assistant (if needed)
register_function(
    search_tool,
    caller=assistant,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

# Instantiate the Teachability capability and add it to the assistant
teachability = teachability.Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)
teachability.add_to_agent(assistant)

# Define LLM configuration for additional agents
llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "timeout": 300,  # Increased timeout to 300 seconds (5 minutes)
    "model": "gpt-3.5-turbo-16k",
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Initialize additional agents for the research paper query task
initializer = UserProxyAgent(
    name="Init",
    code_execution_config=False
)

coder = AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="""You are a Coder. Given a topic, write code to retrieve related papers from the arXiv API, print their title,
    authors, abstract and link. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type.
    The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not
    intended to be executed by the executor. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result.
    Check the execution result returned by the executor. If the result indicates there is an error, fix the error and output the code again.
    Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is
    executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.""",
)

executor = UserProxyAgent(
    name="executor",
    system_message="Executor. Execute the code written by the Coder and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    }
)

scientist = AssistantAgent(
    name="scientist",
    llm_config=llm_config,
    system_message="""You are the Scientist. Please categorize papers after seeing their abstracts printed and create a markdown table with Domain,
    Title, Authors, Summary and Link.""",
)

# Define state transition function for the GroupChat
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is initializer:
        return coder
    elif last_speaker is coder:
        return executor
    elif last_speaker is executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return coder
        else:
            return scientist
    elif last_speaker is scientist:
        return None

# Initialize GroupChat and GroupChatManager
groupchat = autogen.GroupChat(
    agents=[initializer, coder, executor, scientist],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start the interaction with the initial message
initializer.initiate_chat(
    manager, message="Topic: LLM applications papers from last week. Requirement: 5-10 papers.",
)

# Execute the main chat initiation with caching
with Cache.disk(cache_seed=43) as cache:
    initializer.initiate_chat(
        manager,
        message="Topic: LLM applications papers from last week. Requirement: 5-10 papers.",
        cache=cache,
    )
