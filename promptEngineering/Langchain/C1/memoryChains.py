"""The Memory

In LangChain, Memory refers to the mechanism that stores and manages the conversation history between a user
and the AI. It helps maintain context and coherency throughout the interaction, enabling the AI to generate
more relevant and accurate responses.
"""

from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="tell me about Arvind kejriwal.")

# Continue the conversation
conversation.predict(input="What is he doing today")
conversation.predict(input="How can I meet him?")

# Display the conversation
print(conversation)