"""
it manages comma-separated outputs.
 It handles one specific case: anytime you want to receive a list of outputs from the model.
"""

from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word=input("enter word: "),
  context=input("enter context: "),
)

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=input("temperature: "))

# Send the Request
output = model(model_input)
print(parser.parse(output))