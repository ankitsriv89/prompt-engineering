#Using LangChain and Deep Lake to Explore Amazon's Revenue Growth Pre- and Post-Pandemic

#In this tutorial, we will first load Amazon's quarterly financial reports, embed using OpenAI's API,
# store the data in Deep Lake, and then explore it by asking questions.

from dotenv import load_dotenv
import os
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

import textwrap
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
from langchain.document_loaders import PagedPDFSplitter
from langchain.chat_models import ChatOpenAI



embeddings = OpenAIEmbeddings()
# TODO: use your organization id here.  (by default, org id is your username)
my_activeloop_org_id = "ankit"
my_activeloop_dataset_name = "amazon_earnings_6"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding=embeddings, token=activeloop_token)
#db.add_documents(texts)
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff', retriever=db.as_retriever())

print(qa.run("Combined total expenditure in 2020?"))
#Amazon's total revenue in 2020 was $386,064 million.



