"""
Tools in LangChain

LangChain provides a variety of tools for agents to interact with the outside world.
These tools can be used to create custom agents that perform various tasks, such as searching the web,
answering questions, or running Python code.

In our example, two tools are being defined for use within a LangChain agent: a Google Search
tool and a Language Model tool acting specifically as a text summarizer. The Google Search tool,
using the GoogleSearchAPIWrapper, will handle queries that involve finding recent event information.
The Language Model tool leverages the capabilities of a language model to summarize texts. These tools are
designed to be used interchangeably by the agent, depending on the nature of the user's query.

"""

from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
google_api_key = os.environ.get('GOOGLE_API_KEY')
google_cse_id = os.environ.get('GOOGLE_CSE_ID')

from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query = input("Enter your question : ")
#response = agent("What's the latest news about the Chandrayaan 3?")
response = agent(query)
print(response['output'])