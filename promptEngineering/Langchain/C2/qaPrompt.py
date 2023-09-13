"""Creating a Question-Answering Prompt Template"""

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
huggingfacehub_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import HuggingFaceHub, LLMChain

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
#question = input("enter question: ")



# initialize Hub LLM
hub_llm = HuggingFaceHub( repo_id='google/flan-t5-large', model_kwargs={'temperature':0} )

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
#print(llm_chain.run(question))

#Asking Multiple Questions

qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
#res = llm_chain.generate(qa)
#print( res )

llm = OpenAI(model_name="text-davinci-003")

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
    "What color is a ripe banana?\n"
)
print(llm_chain.run(qs_str))

