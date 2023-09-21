
from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_key = os.environ.get("GOOGLE_CSE_ID")

from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType

from langchain.utilities import GoogleSearchAPIWrapper, PythonREPL

search = GoogleSearchAPIWrapper()
python_repl = PythonREPL()

llm = OpenAI(model="text-davinci-003", temperature=0)
toolkit = [
    Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. Useful for saving strings to files.",
        func=python_repl.run
    ),

]

agent = initialize_agent(
    toolkit,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

from langchain.callbacks import get_openai_callback

query = input("ask me a question: ")
with get_openai_callback() as cb:
    agent.run(query)


