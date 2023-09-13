"""
Deep Lake VectorStore

Deep Lake provides storage for embeddings and their corresponding metadata in the context of LLM apps.
It enables hybrid searches on these embeddings and their attributes for efficient data retrieval.
It also integrates with LangChain, facilitating the development and deployment of applications.
"""
from dotenv import load_dotenv
import os

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
# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our documents
texts = [
    "Mohandas Karamchand Gandhi, popularly called Mahatma Gandhi, was born on October 2, 1869. \
    Because of his enormous sacrifices for India, he is sometimes referred to as the Father of the Nation.\
    he was killed on January 30, 1948, in New Delhi.",
    "Born on November 14, 1889, Pandit Jawaharlal Nehru was Motilal Nehru and Swarup Rani Nehru’s child. \
    His birthday, November 14, is widely observed as Children’s Day in India.",
    "Sardar Vallabh Bhai Patel was born on 31 October 1875  and died on 15 December 1950. \
    Often renowned as the Iron Man of India & Bismarck of India, he was famous for his courage and heroism \
    from a very young age. Sardar Patel, who was at first a lawyer, quit his job to battle for India’s \
    freedom from British rule. He was elected India’s Deputy Prime Minister when the country gained independence",
    "Bhagat Singh,a revolutionary hero was born into a Sikh family in the undivided Punjab state and retained \
    his family’s history and nationalism until his death.\
    In 1928, he was implicated in a plan to kill James Scott, a British police superintendent, in retaliation \
    for the death of Lala Lajpat Rai.",
    "Lal Bahadur Shastri was born on October 2, 1904, in Mughalsarai, Uttar Pradesh,  and passed away on January 11, 1966.\
     He was awarded the title of “Shastri” Scholar when he completed his education at Kashi Vidyapeeth.",

]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "ankit"
my_activeloop_dataset_name = "langchain_course_freedom_fighter"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,chain_type="stuff",retriever=db.as_retriever()
)



tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
response = agent.run("When was nehru born?")
print(response)

