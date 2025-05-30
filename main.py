import os
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
import numexpr

# API keys
cohere_api_key = os.environ["COHERE_API_KEY"]
tavili_api_key = os.environ["TAVILY_API_KEY"]

# Initialize Cohere model
chat = ChatCohere(model="command-r-plus", temperature=0.7, api_key=cohere_api_key)

# Tavily Search Tool
internet_search = TavilySearchResults(api_key=tavili_api_key)
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant documents from the internet."

class TavilySearchInput(BaseModel):
    query: str = Field(description="Internet query engine.")

internet_search.args_schema = TavilySearchInput


# Python REPL Tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result.",
    func=python_repl.run,
)

repl_tool.name = "python_interpreter"

class PythonToolInput(BaseModel):
    code: str = Field(description="Python code execution.")

repl_tool.args_schema = PythonToolInput

# File Handler Tool
class FileHandler:
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

file_handler = FileHandler()
file_tool = Tool(
    name="file_handler",
    description="Read or write files. Specify 'read:FILE_PATH' or 'write:FILE_PATH:CONTENT'.",
    func=lambda x: file_handler.read_file(x.split(":")[1]) if x.startswith("read:") else file_handler.write_file(x.split(":")[1], x.split(":", 2)[2]),
)

class FileToolInput(BaseModel):
    command: str = Field(description="File operaiton in format 'read:FILE_PATH' or 'write:FILE_PATH:CONTENT'.")

file_tool.args_chema = FileToolInput

# Wikipedia Search Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool = Tool(
    name="wikipedia_search",
    description="Search Wikipedia for reliable, structured information.",
    func=wikipedia.run,
)

class WikipediaToolInput(BaseModel):
    query: str = Field(description="Query to search on Wikipedia.")

wikipedia_tool.args_chema = WikipediaToolInput

# Math Calculator Tool
class MathCalculator:
    def evaluate(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            result = numexpr.evaluate(expression).item()
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

math_calculator = MathCalculator()
math_tool = Tool(
    name="math_calculator",
    description="Evaluate mathematical expressions.",
    func=math_calculator.evaluate,
)

class MathToolInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate.")

math_tool.args_schema = MathToolInput

# Text prompt that tells the agent to act a certain way.
# Will guide the agent's reasoning.
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
