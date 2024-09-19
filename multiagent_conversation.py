import os
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import UserProxyAgent
import chromadb

llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],
    "temperature": 0
}

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

from autogen import ConversableAgent

user = ConversableAgent(
    name="user",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

fin_aid = RetrieveUserProxyAgent(
    name="FinanceRetrieveAgent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": ["limited_output2.txt", "limited_output.txt",
                      os.path.join(os.path.abspath(""), "..", "website", "docs")],
        "chunk_token_size": 2000,
        "model": "gpt-3.5-turbo",
        "client": chromadb.PersistentClient(path="./fin"),
        "get_or_create": True
    },
    code_execution_config=False,
)

sales_aid = RetrieveUserProxyAgent(
    name="SalesRetrieveAgent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": ["sales_data.csv",
                      os.path.join(os.path.abspath(""), "..", "website", "docs")],
        "chunk_token_size": 2000,
        "model": "gpt-3.5-turbo",
        "client": chromadb.PersistentClient(path="./sales"),
        "get_or_create": True
    },
    code_execution_config=False,
)


finance = AssistantAgent(
    name="Finance_Analyst",
    is_termination_msg=termination_msg,
    system_message="MUST CALL your assigned Function in any case .",
    llm_config=llm_config,
)

sales = AssistantAgent(
    name="Sales_Analyst",
    is_termination_msg=termination_msg,
    system_message="MUST CALL your assigned Function in any case .",
    llm_config=llm_config,
)
summariser = AssistantAgent(
    name="summariser",
    is_termination_msg=termination_msg,
    system_message="question asked by user can have part of finance and sales , you got context from respective agent , just try to given best possible asnswer and with best possible semantics and relationship",
    llm_config=llm_config,
)

verifier = AssistantAgent(
    name="convergence_checker",
    is_termination_msg=termination_msg,
    system_message=" reply 'TERMINATE'",
    llm_config=llm_config,
)
def _reset_agents():
    fin_aid.reset()
    finance.reset()
    sales.reset()
    sales_aid.reset()
from typing_extensions import Annotated
def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call

    # it from other agents.
 
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        sales_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = sales_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and sales_aid.update_context:
            sales_aid.problem = message if not hasattr(sales_aid, "problem") else sales_aid.problem
            _, ret_msg = sales_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = sales_aid.message_generator(sales_aid, None, _context)
        return ret_msg if ret_msg else message
    
    def retrieve_content1(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        fin_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = fin_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and fin_aid.update_context:
            fin_aid.problem = message if not hasattr(fin_aid, "problem") else fin_aid.problem
            _, ret_msg = fin_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = fin_aid.message_generator(fin_aid, None, _context)
        return ret_msg if ret_msg else message


    fin_aid.human_input_mode = "NEVER"  # Disable human input for fin_aid since it only retrieves content.
    sales_aid.human_input_mode = "NEVER"  # Disable human input for sales_aid since it only retrieves content.

    autogen.agentchat.register_function(
    retrieve_content,
    caller=finance,
    executor=user,
    description="retrieve data (RAG) for finance part of user questiom",
)
    autogen.agentchat.register_function(
    retrieve_content1,
    caller=sales,
    executor=user,
    description="retrieve data (RAG) for sales part in question",
)


    groupchat = autogen.GroupChat(
        agents=[user,sales,finance,summariser,verifier],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=True,
    )
    llm_config_manager = llm_config.copy()
    llm_config_manager.pop("functions", None)
    llm_config_manager.pop("tools", None)
    manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_manager,
    is_termination_msg=lambda x: "GROUPCHAT_TERMINATE" in x.get("content", ""),)

    manager = autogen.GroupChatManager(groupchat=groupchat)
    
    
    message = " What types of product we have in our stock or we sold and how we can increase its profit ?"
    user.initiate_chat(manager,
        message=message)

call_rag_chat()