"""
This class uses the Pydantic library, which helps define and validate data structures in Python.
It enables us to characterize the expected output with a name, type, and description. We need a variable
that can store multiple suggestions in the thesaurus example.
It can be easily done by defining a class that inherits from the Pydantic’s BaseModel class.
"""
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')

import tkinter as tk
from tkinter import Label, Entry, Button, Text, Scrollbar
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

    # Throw error in case of receiving a numbered-list from API
    @field_validator('words')
    @classmethod
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

    @field_validator('reasons')
    @classmethod
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

#template = """
#Offer a list of suggestions to substitue the specified target_word based the presented context.
#{format_instructions}
#target_word={target_word}
#context={context}
#"""

# Create a Tkinter window
window = tk.Tk()
window.title("Suggestions Generator")
window.geometry("600x400")  # Set the window size

# Create and configure input fields and labels
target_word_label = Label(window, text="Target Word:")
target_word_label.pack()
target_word_entry = Entry(window)
target_word_entry.pack()

context_label = Label(window, text="Context:")
context_label.pack()
context_entry = Text(window, height=5, width=40)
context_entry.pack()

temperature_label = Label(window, text="Temperature for AI Creativity:")
temperature_label.pack()
temperature_entry = Entry(window)
temperature_entry.pack()

# Create and configure a response text box with scrollbar
response_text = Text(window, height=10, width=60)
response_text.pack()
scrollbar = Scrollbar(window, command=response_text.yview)
scrollbar.pack(side="right", fill="y")
response_text.config(yscrollcommand=scrollbar.set)


# Function to execute the code when the "Run" button is clicked
def execute_code():
    target_word = target_word_entry.get()
    context = context_entry.get("1.0", "end-1c")
    temperature = float(temperature_entry.get())

    # Define your template and prompt here (same as in your code)
    template = """
    Offer a list of suggestions to substitute the specified target_word based on the presented context and 
    the reasoning for each word.
    {format_instructions}
    target_word={target_word}
    context={context}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["target_word", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model_input = prompt.format_prompt(target_word=target_word, context=context)

    # Initialize the OpenAI model (same as in your code)
    #temperature = float(input("enter temperature of AI creativity: "))
    model = OpenAI(model_name='text-davinci-003', temperature=temperature)

    # Generate the response
    output = model(model_input.to_string())

    # Parse and display the response
    response_text.delete("1.0", "end")
    response_text.insert("1.0", parser.parse(output))

"""
The temperature value could be anything between 0 to 1, where a higher number means the model is more creative. 
Using larger value in production is a good practice if you deal with tasks requiring creative output.
"""



run_button = Button(window, text="Run", command=execute_code)
run_button.pack()

# Start the Tkinter main loop
window.mainloop()