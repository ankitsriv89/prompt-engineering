from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
#google_cse_id = os.environ.get('GOOGLE_CSE_ID')

llm = OpenAI(model="text-davinci-003", temperature=0.9)

text = "suggest a way to regrow lost hairs"
print(llm(text))
