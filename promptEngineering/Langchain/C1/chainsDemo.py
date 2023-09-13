"""The Chains

In LangChain, a chain is an end-to-end wrapper around multiple individual components,
providing a way to accomplish a common use case by combining these components in a specific sequence.
The most commonly used type of chain is the LLMChain, which consists of a PromptTemplate,
a model (either an LLM or a ChatModel), and an optional output parser.

The LLMChain works as follows:

Takes (multiple) input variables.
Uses the PromptTemplate to format the input variables into a prompt.
Passes the formatted prompt to the model (LLM or ChatModel).
If an output parser is provided, it uses the OutputParser to parse the output of the LLM into a final format.

By using LangChain's LLMChain, PromptTemplate, and OpenAIclasses, we can easily define our prompt,
set the input variables, and generate creative outputs.
"""

from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest three good names for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("interplanetary space travel"))