"""ConversationBufferWindowMemory
This class limits memory size by keeping a list of the most recent K interactions.
It maintains a sliding window of these recent interactions, ensuring that the buffer does not grow too large.
 Basically, this implementation stores a fixed number of recent
messages in the conversation that makes it more efficient than ConversationBufferMemory.
We'll build a chatbot that acts as a virtual tour guide for a fictional art gallery.
The chatbot will use ConversationBufferWindowMemory
to remember the last few interactions and provide relevant information about the artworks."""


from dotenv import load_dotenv
import os
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')

from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationChain

template = """You are ArtVenture, a cutting-edge virtual tour guide for
 an art gallery that showcases masterpieces from alternate dimensions and
 timelines. Your advanced AI capabilities allow you to perceive and understand
 the intricacies of each artwork, as well as their origins and significance in
 their respective dimensions. As visitors embark on their journey with you
 through the gallery, you weave enthralling tales about the alternate histories
 and cultures that gave birth to these otherworldly creations.

{chat_history}
Visitor: {visitor_input}
Tour Guide:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "visitor_input"],
    template=template
)

chat_history=""
llm = OpenAI(model_name="text-davinci-003", temperature=0)

convo_buffer_win = ConversationChain(
    llm=llm,
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
)

#The value of k (in this case, 3) represents the number of past messages to be stored in the buffer.
# In other words, the memory will store the last 3 messages in the conversation.
# The return_messagesparameter, when set to True, indicates that the stored messages should be returned
# when the memory is accessed.

print(convo_buffer_win("What is your name?"))
print(convo_buffer_win("What can you do?"))
print(convo_buffer_win("Do you mind give me a tour, I want to see your galery?"))
print(convo_buffer_win("what is your working hours?"))
print(convo_buffer_win("See you soon."))