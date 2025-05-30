# LangChain Chatbot Agent
This project is a versatile chatbot agent built with LangChain, powered by Cohere's command-r-plus model. It integrates multiple tools to handle a variety of tasks, including internet searches, secure Python code execution, file operations, Wikipedia queries, and mathematical calculations. The agent is designed for secure and efficient processing of complex queries, with a focus on data analysis and visualization.

## Features

**Internet Search**: Uses Tavily to fetch relevant web content for up-to-date information.
**Secure Python Execution**: Executes Python code in a sandboxed environment using restrictedpython, supporting math, numpy, and matplotlib.pyplot for safe computations and visualizations.
**File Handling**: Reads and writes files (e.g., text, CSV) in a secure output directory.
**Wikipedia Search**: Queries Wikipedia for reliable, structured information.
**Math Calculations**: Evaluates mathematical expressions safely using numexpr.
**Modular Design**: Built with LangChain's ReAct framework for multi-step reasoning and tool selection.

## Prerequisites

- Python 3.8 or higher
- Conda environment (recommended)
- API keys for:
- Cohere (COHERE_API_KEY)
- Tavily (TAVILY_API_KEY)



## Installation

1. Clone the Repository (if applicable):
```
git clone <repository-url>
cd <repository-directory>
```


2. Create a Conda Environment:
```
conda create -n chatbot_agent python=3.8
conda activate chatbot_agent
```

3. Install Dependencies:
```
conda install -c conda-forge restrictedpython matplotlib numpy
pip install langchain langchain-cohere langchain-community
```

4. Set Environment Variables: Set the keys in your conda environment
```
conda env config vars set COHERE_API_KEY=your_cohere_key
conda env config vars set TAVILY_API_KEY=your_tavily_key
```


## Usage

Run the Agent:Save the main script as chatbot_app.py and execute:
python chatbot_app.py


@@ Example Queries:

Mathematical Calculation:response = agent_executor.invoke({"input": "Calculate 2^10"})
Expected output: "1024"


Wikipedia Search:response = agent_executor.invoke({"input": "Search Wikipedia for the history of Python programming"})
Returns a summary of Python's history


Python Visualization:response = agent_executor.invoke({"input": "Create a pie chart of data [1, 2, 3, 4] using matplotlib"})
Saves chart to 'output/plot.png'


File Operation:response = agent_executor.invoke({"input": "Save 'Hello' to 'test.txt'"})
Saves to 'output/test.txt'



## Security Features

**Secure Python REPL**: Uses restrictedpython to sandbox code execution, allowing only safe modules (math, numpy, matplotlib.pyplot). Dangerous modules like os, sys, and subprocess are blocked.
**File Handling**: Restricts file operations to the output directory to prevent unauthorized access.
**Input Validation**: Tools use Pydantic schemas to ensure structured and safe inputs.
**Error Handling**: Gracefully handles errors from API calls, file operations, and code execution.

## Project Structure

**chatbot_app.py**: Main script containing the agent and tool definitions.
**output/**: Directory for file outputs (e.g., charts, text files).

## Limitations

Requires internet access for Tavily and Cohere API calls.
Python REPL is limited to safe modules; additional modules can be added to safe_globals_dict if needed.
File operations are restricted to the output directory for security.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bug reports or feature requests.

## Contact
For questions or support, contact the project maintainer or open an issue on the repository.
