
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
huggingfacehub_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

summarization_template = "Summarize the following text to two sentence: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

text = input("enter text to summarize : ")
summarized_text = summarization_chain.predict(text=text)
print(summarized_text)