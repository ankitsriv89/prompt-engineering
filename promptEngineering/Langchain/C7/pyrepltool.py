from dotenv import load_dotenv
import os
import textwrap

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
serp_api_key = os.environ.get('SERPAPI_KEY')
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_key = os.environ.get("GOOGLE_CSE_ID")
# As a standalone utility:
from langchain. utilities import GoogleSearchAPIWrapper
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType
import tkinter as tk
from tkinter import Entry, Button, Text, Scrollbar

llm = OpenAI(model_name="text-davinci-003", temperature=0)
tools = load_tools(["python_repl"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
  )

def run_query():
    query_text = query_entry.get()
    result = agent(query_text)
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, result)
    result_text.config(state=tk.DISABLED)

# Create the main application window
app = tk.Tk()
app.title("AI Query Agent")



# Create and pack GUI components
query_label = tk.Label(app, text="Enter your search query:")
query_label.pack(pady=10)

query_entry = Entry(app, width=40)
query_entry.pack()

query_button = Button(app, text="Search", command=run_query)
query_button.pack(pady=10)

result_text = Text(app, wrap=tk.WORD, width=50, height=10)
result_text.pack(padx=20, pady=10)

scrollbar = Scrollbar(app, command=result_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.config(yscrollcommand=scrollbar.set)

result_text.config(state=tk.DISABLED)

# Start the Tkinter main loop
app.mainloop()