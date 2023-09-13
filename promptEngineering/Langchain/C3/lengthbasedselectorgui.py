
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')


from langchain.llms import OpenAI
from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!"
    }, {
        "query": "What is the speed of light?",
        "answer": "Fast enough to make a round trip around Earth 7.5 times in one second!"
    }, {
        "query": "What is a quantum computer?",
        "answer": "A magical box that harnesses the power of subatomic particles to solve complex problems."
    }, {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'."
    }, {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite."
    }, {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes."
    }, {
        "query": "What is photosynthesis?",
        "answer": "A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'"
    }, {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump."
    }, {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies."
    }, {
        "query": "What is the largest mammal on Earth?",
        "answer": "The blue whale, the original heavyweight champion of the world."
    }, {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom."
    }, {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette."
    }, {
        "query": "What is the average temperature on Mars?",
        "answer": "Cold enough to make a Martian wish for a sweater and a hot cocoa."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

import tkinter as tk
from tkinter import Text, Scrollbar


# Function to execute the code when the "Run" button is clicked
def execute_code():
    query = input_entry.get("1.0", "end-1c")
    input_data = {"query": query}

    # Place your code here (from line 80 onwards) to run the LLMChain
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=100
    )

    dynamic_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n"
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # Existing example and prompt definitions, and dynamic_prompt_template initialization

    # Create the LLMChain for the dynamic_prompt_template
    chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)
    response = chain.run(input_data)

    response_text.delete("1.0", "end")
    response_text.insert("1.0", response)


# Create the main window
window = tk.Tk()
window.title("LangChain Humour AI")
window.geometry("600x400")  # Set the window size

# Create a colorful background
window.configure(bg="lightblue")

# Create and configure the input text box
input_label = tk.Label(window, text="Enter your query:")
input_label.pack()
input_entry = Text(window, height=5, width=40)
input_entry.pack()

# Create a "Run" button to execute the code
run_button = tk.Button(window, text="Run", command=execute_code)
run_button.pack()

# Create and configure the response text box with scrollbar
response_text = Text(window, height=10, width=40)
response_text.pack()
scrollbar = Scrollbar(window, command=response_text.yview)
scrollbar.pack(side="right", fill="y")
response_text.config(yscrollcommand=scrollbar.set)

# Start the Tkinter main loop
window.mainloop()
