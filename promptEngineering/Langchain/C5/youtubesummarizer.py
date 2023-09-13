# -*- coding: utf-8 -*-
"""youtubeSummarizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_qDdeTgzz2p39J6OnLVcSoVpOdX-tfXj
"""

import os
import yt_dlp

def download_mp4_from_youtube(url):
    # Set the options for the download
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
download_mp4_from_youtube(url)

import whisper

model = whisper.load_model("base")
result = model.transcribe("lecuninterview.mp4")
print(result['text'])

with open ('text.txt', 'w') as file:
    file.write(result['text'])

#Summarization with LangChain

from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="text-davinci-003", temperature=0)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

from langchain.docstore.document import Document

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]

from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

#With the following line of code, we can see the prompt template that is used with the map_reduce technique.
print( chain.llm_chain.prompt.template )

#The "stuff" approach is the simplest and most naive one, in which all the text from the transcribed video is used in a single prompt.
#This method may raise exceptions if all text is longer than the available context size of the LLM and may not be the most efficient way
#to handle large amounts of text.

#below. This prompt will output the summary as bullet points.

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["text"])

#initialized the summarization chain using the stuff as chain_type and the prompt above

chain = load_summarize_chain(llm,
                             chain_type="stuff",
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary,
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)

#The 'refine' summarization chain is a method for generating more accurate and context-aware summaries.
#This chain type is designed to iteratively refine the summary by providing additional context when needed.
#That means: it generates the summary of the first chunk. Then, for each successive chunk,
#the work-in-progress summary is integrated with new info from the new chunk.

chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

#Adding Transcripts to Deep Lake

#This method can be extremely useful when you have more data.
#Let’s see how we can improve our expariment by adding multiple URLs,
#store them in Deep Lake database and retrieve information using QA chain.

#modify the script for video downloading slightly, so it can work with a list of URLs.

import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

urls=["https://youtu.be/92jMPf9IaZU",
      "https://youtu.be/SVa66vO08So",
      "https://youtu.be/_TnPGeefpWc",]
vides_details = download_mp4_from_youtube(urls, 1)

#transcribe the videos using Whisper as we previously saw and save the results in a text file.

import whisper

# load the model
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in vides_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    #print(f"Transcription for {video[0]}:\n{result['text']}\n")

# Open the file and write the transcriptions
with open('text.txt', 'w') as file:
    for transcription in results:
        file.write(transcription + '\n')

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the texts
with open('text.txt') as f:
    text = f.read()
texts = text_splitter.split_text(text)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
texts = text_splitter.split_text(text)

#pack all the chunks into a Documents:
from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]

# import Deep Lake and build a database with embedded documents
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "ankit"
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

"""The distance metric determines how the Retriever measures "distance" or similarity between
different data points in the database. By setting distance_metric to 'cos', the Retriever will use
cosine similarity as its distance metric. Cosine similarity is a measure of similarity between two
non-zero vectors of an inner product space that measures the cosine of the angle between them.
It's often used in information retrieval to measure the similarity between documents or pieces
of text. Also, by setting 'k' to 4, the Retriever will return the 4 most similar or closest
results according to the distance metric when a search is performed."""

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#Lastly, we can use the chain_type_kwargs argument to define the custom prompt and for chain type the
#‘stuff’  variation was picked.

from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)
query = input("enter your query: ")

print(qa.run(query))

