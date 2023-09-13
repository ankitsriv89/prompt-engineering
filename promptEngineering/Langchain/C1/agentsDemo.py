"""Agents in LangChain

In LangChain, agents are high-level components that use language models (LLMs) to determine which actions
to take and in what order. An action can either be using a tool and observing its output or returning it to
the user. Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an
Observation, and repeating that until done.

In our example, the Agent will use the Google Search tool to look up recent information about the Mars
rover and generates a response based on this information.
"""
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')
google_cse_id = os.environ.get('GOOGLE_CSE_ID')

from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

llm = OpenAI(model="text-davinci-003", temperature=0)


# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=6)

query = input("Enter your question : ")
#response = agent("What's the latest news about the Chandrayaan 3?")
response = agent(query)
print(response['output'])