
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import OpenAI, ConversationChain

import textwrap
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = OpenAI(model_name="text-davinci-003", temperature=0)


memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


print( conversation.predict(input="Tell me a joke about elephants") )
print( conversation.predict(input="Who is the author of the Harry Potter series?") )
print( conversation.predict(input="What was the joke you told me earlier?") )