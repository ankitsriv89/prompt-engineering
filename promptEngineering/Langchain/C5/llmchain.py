"""
Several methods are available for utilizing a chain, each yielding a distinct output format.
The example in this section is creating a bot that can suggest a replacement word based on context.
"""
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "provide 5 alternative words to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="text-davinci-003", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
word = input("enter word: ")
print(llm_chain(word))

#use the .apply() method to pass multiple inputs at once and receive a list for each input.
"""input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]"""

input_list = [{"word": word} for word in input("Enter words separated by spaces (press Enter to finish): ").split()]
print(llm_chain.apply(input_list))

#The .generate() method will return an instance of LLMResult, which provides more information.
llm_chain.generate(input_list)

#.predict(). (which could be used interchangeably with .run()) Its best use case is to pass multiple inputs
# for a single prompt.
# However, it is possible to use it with one input variable as well.

prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

print(llm_chain.predict(word=input("word for context: "), context=input("context: ")))
# or llm_chain.run(word="fan", context="object")