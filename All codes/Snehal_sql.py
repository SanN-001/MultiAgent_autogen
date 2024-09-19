import pandas as pd
import numpy as np
from langgraph.prebuilt import create_react_agent
file_path = 'pup.csv'  
df = pd.read_csv(file_path)


new_data_point = np.array([	96757.08,27873.3,5906,10550.4,8144.78,5789.75,6941])
print("user" "-->",new_data_point)

def find_closest_data_point_in_df(df, new_data_point):
    min_distance = float('inf')
    closest_index = -1
    
    for index, row in df.iterrows():
        point = np.array(row[['Total revenue',	'Beverage',	'Condiments','Confections','Seafood','Produce','Dairy_Products']]) 
        distance = np.linalg.norm(point - new_data_point)  
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    
    return closest_index

closest_index = find_closest_data_point_in_df(df, new_data_point)

performance_columns = ['Beverage_performance', 'Condiments_performance', 
                       'Confections_performance', 'Seafood_performance', 
                       'Produce_performance', 'Dairy_performance', 
                       'Revenue_performance']

concatenated_values = []


for col in performance_columns:
    concatenated_values.append(f"{col}: {df[col][closest_index]}")

concatenated_string = ', '.join(concatenated_values)


import chromadb
from typing_extensions import Annotated
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os;

llm_config = {
    "timeout": 600,
    "cache_seed": 44,  # change the seed for different trials
    "config_list": [{"model": "gpt-4", "api_key": "sk-proj-nNevbQswbaqu3gnYhrRTT3BlbkFJdcPUak4NBUtG8XP6iFWI"}],
    "temperature": 0
}
from autogen import AssistantAgent, UserProxyAgent

proxy = UserProxyAgent(
    "user_proxy",human_input_mode="NEVER")
pf = AssistantAgent(
    name="problem_indentifier",
    system_message="""You are Senior sales analyst.Given sales manager persona as query ,use ***(*)** as delimeter for different lines. generate questions based on problems 4-5 impactful strictly based on given persona{concatenated_string} in ***system message problems***(short and coincise) ?
     reference->>
      Schema |          Name          | Type  | Owner
--------+------------------------+-------+-------
 public | categories             | table | admin
 CategoryID	CategoryName	Description
 public | customers              | table | admin
 CustomerID	CustomerName	ContactName	Address	City	PostalCode	Country
 public | employees              | table | admin
 EmployeeID	LastName	FirstName	BirthDate	Photo	Notes
 public | order_details          | table | admin
 OrderDetailID	OrderID	ProductID	Quantity
 public | orders                 | table | admin
 OrderID	CustomerID	EmployeeID	OrderDate	ShipperID
public | product                | table | admin
ProductID	ProductName	SupplierID	CategoryID	Unit	Price """
    ,
    llm_config=llm_config,
    human_input_mode="NEVER",
)
persona_description = """
persona of sales manager  ,
"""
persona_description+= concatenated_string 

pf.reset()
proxy.initiate_chat(
    pf,message=concatenated_string,max_turns=1
)
a=""
a+=pf.last_message()["content"]
questions = a.split("*")

# Initialize an empty list to store dictionaries
arr = []

# Convert each question into a dictionary and append to 'arr'
for idx, question in enumerate(questions, start=1):
    if question.strip():  # Ensure not to include empty strings
        problem_dict = {question.strip()}  # Create dictionary
        arr.append(problem_dict)  # Append to list


import time

def countdown(seconds):
    while seconds:
        mins, secs = divmod(seconds, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        seconds -= 1

    print("Time's up!")



from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///nw2.db")
db.text_factory = lambda x: str(x, 'utf-8', 'ignore') if isinstance(x, bytes) else str(x, 'utf-8', 'ignore')
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", openai_api_key = "sk-proj-nNevbQswbaqu3gnYhrRTT3BlbkFJdcPUak4NBUtG8XP6iFWI",temperature=0.7)
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()
from langchain_core.messages import SystemMessage

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent



from autogen import AssistantAgent, UserProxyAgent

proxy = UserProxyAgent(
    "user_proxy",human_input_mode="NEVER")
p = AssistantAgent(
    name="Analyzer",
    system_message="""You are Senior sales .filter sales related content  ,generate **Insights** and summarise more quantative ****not general but like expert ***in short ."""
    ,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


for i in range(0,len(arr)):
    agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)
    chunks = []
    ch=""
    for s in agent_executor.stream(
        {"messages": [HumanMessage(content=str(arr[i]))]}):
        chunks.append(s)
        countdown(10)
        print("query1_performed:")
    for chunk in chunks:
        if "tools" in chunk:
            ch += str(chunk["tools"]['messages'][0])
        elif "agent" in chunk:
            ch += str(chunk["agent"]['messages'][0])
    proxy.initiate_chat(p,message=ch,max_turns=1)
    print(ch)
    print("query2_performed:")
    countdown(100)
    





