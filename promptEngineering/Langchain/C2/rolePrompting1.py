"""Role prompting involves asking the LLM to assume a specific role or identity before performing a
given task, such as acting as a copywriter.
This can help guide the model's response by providing a context or perspective for the task.

In this example, the LLM is asked to act as a
futuristic robot band conductor and suggest a song title related to the given theme and year.
"""

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Create the LLMChain for the prompt
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Input data for the prompt
theme, year = input("enter theme, year for the song in the format: ").split(",")
input_data = {"theme": theme, "year": year}

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)