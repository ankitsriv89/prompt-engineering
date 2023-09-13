"""The purpose of this lesson is to enhance our previous implementation of a News Article Summarizer.
 Our objective is to make our tool even more effective at distilling key information from lengthy news articles and presenting that information in an easy-to-digest, bulleted list format. This enhancement will enable users to quickly comprehend the main points of an article in a clear,
  organized way, thus saving valuable time and enhancing the reading experience."""
import time

from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

import requests
from newspaper import Article
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from pydantic import validator
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import ( HumanMessage )


def fetch_article(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching article from {url}: {e}")
    except Exception as e:
        raise Exception(f"Error parsing article: {e}")

def summarize_article(article_title, article_text, temperature):
    # prepare template for prompt
    template = """
            As an advanced AI, you've been tasked to summarize online articles into bulleted points. 
            Here are a few examples of how you've done this in the past:

            Example 1:
            Original Article: 'The Effects of Climate Change
            Summary:
            - Climate change is causing a rise in global temperatures.
            - This leads to melting ice caps and rising sea levels.
            - Resulting in more frequent and severe weather conditions.

            Example 2:
            Original Article: 'The Evolution of Artificial Intelligence
            Summary:
            - Artificial Intelligence (AI) has developed significantly over the past decade.
            - AI is now used in multiple fields such as healthcare, finance, and transportation.
            - The future of AI is promising but requires careful regulation.

            Now, here's the article you need to summarize:

            ==================
            Title: {article_title}

            {article_text}
            ==================

            Please provide a summarized version of the article in a bulleted list format.
            
            {format_instructions}
            """

    # Format the Prompt
    #prompt = template.format(article_title=article_title, article_text=article_text)

    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["article_title", "article_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    # Format the prompt using the article title and text obtained from scraping
    formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)
    model = OpenAI(model_name="text-davinci-003", temperature=temperature)

    # Use the model to generate a summary
    output = model(formatted_prompt.to_string())

    # Parse the output into the Pydantic model
    parsed_output = parser.parse(output)
    return parsed_output

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @field_validator('summary')
    @classmethod
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

if __name__ == "__main__":
    try:
        article_url = input("Enter article URL: ")
        temperature = float(input("Enter temperature of AI creativity: "))

        article_title, article_text = fetch_article(article_url)
        summary = summarize_article(article_title, article_text, temperature)

        print(summary)
    except Exception as e:
        print(f"An error occurred: {e}")
