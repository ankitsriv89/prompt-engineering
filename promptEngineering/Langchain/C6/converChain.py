from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')

import tkinter as tk

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import OpenAI, ConversationChain

import textwrap

llm = OpenAI(model_name="text-davinci-003", temperature=0)

memory = ConversationBufferMemory(return_messages=True)

while True:
    input_text = input("Enter your query (type 'no' to exit): ")
    if input_text.lower() == 'no':
        break

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    response = conversation.predict(input=input_text)
    print(textwrap.fill(response, width=100))
    #print(response)
