from dotenv import load_dotenv
import os

from langchain.agents import Tool

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


#llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# load the existing Deep Lake dataset and specify the embedding function
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "ankit"
my_activeloop_dataset_name = "langchain_course_freedom_fighter"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Napoleon was born on the island of Corsica to a native family descending from Italian nobility."
    "He supported the French Revolution in 1789 while serving in the French army",
    "The British exiled Napolean to the remote island of Saint Helena in the Atlantic, where he died in 1821 at the age of 51.",
    "Napoleon's family was of Italian origin.",
    "Mia Malkova is an American pornographic actress and media personality.[2][3] She has won several awards, including the 2014 AVN Award for the Best New Starlet and the 2017 XBIZ Award for the Best Actress.",
    "Malkova has also acted in an Indian pornographic short monologue documentary and an Indian erotic thriller film, both directed by Ram Gopal Varma"
    "Ram Prasad Bismil was hanged to death at the Gorakhpur Jail.",
    "Mangal Pandey was Born in 1827 "
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# instantiate the wrapper class for GPT3
llm = OpenAI(model="text-davinci-003", temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=db.as_retriever())

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent( tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run("Which documentary Mia Malkova acted in? and when did mangal pandey born")
print(response)