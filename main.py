import os
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
import numexpr

print("Current working directory:", os.getcwd())

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
            # Ensure output directory exists
            if not file_path.startswith("output" + os.sep):
                output_path = os.path.join("output", file_path)
            else:
                output_path = file
            with open(output_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
        
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            os.makedirs("output", exist_ok=True) # Ensure output directory exists
            
            # Check if the file path starts with "output" to avoid double nesting
            if not file_path.startswith("output" + os.sep):
                output_path = os.path.join("output", file_path)
            else:
                output_path = file_path
            with open(output_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

file_handler = FileHandler()

class FileToolInput(BaseModel):
    operation: str = Field(description="Operation to perform on the file, either 'read' or 'write'.")
    file_path: str = Field(description="Path to the file.")
    content: str = Field(default="", description="Content to write to the file (if operation is 'write').")

def file_tool_func(input: FileToolInput):
    if input.operation == "read":
        return file_handler.read_file(input.file_path)
    elif input.operation == "write":
        return file_handler.write_file(input.file_path, input.content)
    else: 
        return "Invalid operation. Use 'read' or 'write'."

file_tool = StructuredTool(
    name="file_handler",
    description="Read or write files. Use operation='read' or 'write', specify file path, and content for writing.",
    func=file_tool_func,
    args_schema=FileToolInput,
)

file_tool.args_schema = FileToolInput

# Wikipedia Search Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool = Tool(
    name="wikipedia_search",
    description="Search Wikipedia for reliable, structured information.",
    func=wikipedia.run,
)

class WikipediaToolInput(BaseModel):
    query: str = Field(description="Query to search on Wikipedia.")

wikipedia_tool.args_schema = WikipediaToolInput

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
prompt = ChatPromptTemplate.from_template(
    """You are a versatile assistant capable of searching the internet, executing Python code, reading/writing files, searching Wikipedia, and performing mathematical calculations.
    Break down complex queries into steps, use the appropriate tool for each step, and provide clear, concise answers.
    For visualizations, use the python_interpreter with Matplotlib.
    For file operations, use file_handler with operation='read' or 'write', specify file_path, and content for writing.
    Use wikipedia_search for factual or historical queries and math_calculator for precise calculations.
    Input: {input}"""
)

# Create the agent with the tools and prompt
agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, repl_tool, file_tool, wikipedia_tool, math_tool],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search, repl_tool, file_tool, wikipedia_tool, math_tool], verbose=True)

response = agent_executor.invoke({
    "input": "Search Wikipedia for the history of Python programming, calculate 2^10, and save the Wikipedia summary to a file named 'python_history.txt'.",
})
