import tkinter as tk
from tkinter import scrolledtext
from dotenv import load_dotenv
import os
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

load_dotenv()


# Your code for examples and setup remains the same
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)



# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
def run_language_model():
    query = input_text.get("1.0", "end-1c")
    response = chain.run(query)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, response)
    output_text.config(state=tk.DISABLED)
    output_text.config(bg='black', fg='white')

# Create the main GUI window
window = tk.Tk()
window.title("LangChain Chat")
window.geometry("600x400")

# Create a label
label = tk.Label(window, text="Enter your question:")
label.pack(pady=10)

# Create an input text box
input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=5)
input_text.pack()

# Create a button to run the language model
run_button = tk.Button(window, text="Run Language Model", command=run_language_model)
run_button.pack(pady=10)

# Create a label for the output
output_label = tk.Label(window, text="Response:")
output_label.pack()

# Create an output text area
output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=40, height=10)
output_text.pack()

output_text.config(state=tk.DISABLED)  # Disable the output text area initially

# Create the LangChain instance and load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)

window.mainloop()
