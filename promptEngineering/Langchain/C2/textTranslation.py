from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
huggingfacehub_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

source_language = input("Enter source language: ")
target_language = input("Enter target language: ")
text = input("enter text to translate: ")
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
print(translated_text)



