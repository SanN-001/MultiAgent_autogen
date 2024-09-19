import os
import autogen

from autogen import ConversableAgent

# Ensure the API key is set correctly
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

# Initialize the ConversableAgent with gpt-3.5-turbo-16k model
# agent = ConversableAgent(
#     "chatbot",
#     llm_config={"config_list": [{"model": "gpt-3.5-turbo-16k", "api_key": api_key}]},
#     code_execution_config=False,  # Turn off code execution, by default it is off.
#     function_map=None,  # No registered functions, by default it is None.
#     human_input_mode="NEVER",  # Never ask for human input.
# )

# Generate a reply
# reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
# print(reply)

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    #"config_list": config_list,
    "timeout": 120,
    "model": "gpt-3.5-turbo-16k",
    "api_key": api_key
}

initializer = autogen.UserProxyAgent(
    name="Init",
    code_execution_config=False
)

coder = autogen.AssistantAgent(
     name="coder",
     llm_config=llm_config,
     system_message="""You are a Coder. Given a topic, write code to retrive related papers from the arXiv API, print their title,
     authors, abstract and link. YOu write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type.
     The user can't modify your code. SO do not suggest incomplete code which requires others to modify. Don't use a code blok if it's not
     intended to be executed by the executor. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result.
     Check the execution result returned by the executor. If the result indicates there is an error, fix the error and output - the code again.
     Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is
     executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.""",
)

executor = autogen.UserProxyAgent(
     name = "executor",
     system_message="Executor. Execute the code written by the Coder and report the result.",
     human_input_mode="NEVER",
     code_execution_config={
         "last_n_messages": 3,
         "work_dir": "paper",
         "use_docker": False,
     }
)

scientist = autogen.AssistantAgent(
     name="scientist",
     llm_config=llm_config,
     system_message="""You are the Scientist. Please categorize papers after seeing their abstracts printed and create markdown table with Domain,
     Title, Authors, SUmmary and Link""",
)

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

groupchat = autogen.GroupChat(
    agents=[initializer, coder, executor, scientist],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

initializer.initiate_chat(
    manager, message="Topic: LLM applications papers from last week. Requirement: 5-10 papers.",
)