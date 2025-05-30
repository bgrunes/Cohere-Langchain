import os
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults

cohere_api_key = os.environ["COHERE_API_KEY"]
chat = ChatCohere(model="command-r-plus", temperature=0.7, api_key=cohere_api_key)

internet_search = TavilySearchResults(api_key=os.environ["TAVILY_API_KEY"])
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant documents from the internet."

from pydantic import BaseModel, Field

class TavilySearchInput(BaseModel):
    query: str = Field(description="Internet query engine.")

internet_search.args_schema = TavilySearchInput

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result.",
    func=python_repl.run,
)

repl_tool.name = "python_interpreter"

class ToolInput(BaseModel):
    code: str = Field(description="Python code execution.")

repl_tool.args_schema = ToolInput

from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool], verbose=True)

response = agent_executor.invoke({
    "input": "Create a pie chart of the top 5 most used programming languages in 2025.",
})