from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model_name="text-davinci-003", temperature=0)

word = input("enter word: ")
output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for '{word}' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=["word"]),
    output_parser=output_parser)

print(llm_chain.predict(word=word))

#Conversational Chain (Memory)
#LangChain provides a ConversationalChain to track previous prompts and responses using the
# ConversationalBufferMemory class.

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

print(conversation.predict(input="List all possible words as substitute for '{word}' as comma separated."))
print(conversation.predict(input="And the next 4?"))

"""It is possible to trace the inner workings of any chain by setting the verbose argument to True."""
