from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

from transformers import AutoTokenizer

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
f = open("tokens.txt",'w')
print(tokenizer.vocab,file=f)

token_ids = tokenizer.encode(input("Enter Text: "))

print( "Tokens:   ", tokenizer.convert_ids_to_tokens( token_ids ) )
print( "Token IDs:", token_ids )