"""This lesson will explore the idea of finding the best articles from the Internet as the
context for a chatbot to find the correct answer. We will use LangChain’s integration with Google
Search API and the Newspaper library to extract the stories
from search results. This is followed by choosing and using the most relevant options in the prompt."""

from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
serp_api_key = os.environ.get('SERPAPI_KEY')
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_key = os.environ.get("GOOGLE_CSE_ID")

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import newspaper

search = GoogleSearchAPIWrapper()
TOP_N_RESULTS = 10

def top_n_results(query):
    return search.results(query, TOP_N_RESULTS)

tool = Tool(
    name = "Google Search",
    description="Search Google for recent results.",
    func=top_n_results
)

query = input("ask me a question: ")

results = tool.run(query)

for result in results:
    print(result["title"])
    print(result["link"])
    print(result["snippet"])
    print("-"*50)



pages_content = []

for result in results:
    try:
        article = newspaper.Article(result["link"])
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({ "url": result["link"], "text": article.text })
    except:
        continue

#Process the Search Results
# We now have the top 10 results from the Google search.
# However, it is not efficient to pass all the contents to the model, So, let’s find the most relevant results,
#
# Incorporating the LLMs embedding generation capability will enable us to find contextually similar content.
# It means converting the text to a high-dimensionality tensor that captures meaning.
# The cosine similarity function can find the closest article with respect to the user’s question.


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

docs = []
for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata={ "source": d["url"] })
        docs.append(new_doc)



embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

docs_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
query_embedding = embeddings.embed_query(query)

#Lastly, the get_top_k_indices function accepts the content and query embedding vectors and returns the
# index of top K candidates with the highest
# cosine similarities to the user's request. Later, we use the indexes to retrieve the best-fit documents.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
  # convert the lists of vectors to numpy arrays
  list_of_doc_vectors = np.array(list_of_doc_vectors)
  query_vector = np.array(query_vector)

  # compute cosine similarities
  similarities = cosine_similarity(query_vector.reshape(1, -1), list_of_doc_vectors).flatten()

  # sort the vectors based on cosine similarity
  sorted_indices = np.argsort(similarities)[::-1]

  # retrieve the top K indices from the sorted list
  top_k_indices = sorted_indices[:top_k]

  return top_k_indices

top_k = 2
best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
best_k_documents = [doc for i, doc in enumerate(docs) if i in best_indexes]

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

response = chain({"input_documents": best_k_documents, "question": query}, return_only_outputs=True)

response_text, response_sources = response["output_text"].split("SOURCES:")
response_text = response_text.strip()
response_sources = response_sources.strip()

print(f"Answer: {response_text}")
print(f"Sources: {response_sources}")

