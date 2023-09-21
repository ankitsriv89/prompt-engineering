
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
serp_api_key = os.environ.get('SERPAPI_KEY')

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI
import textwrap
import tkinter as tk
from tkinter import Entry, Button, Label,Text, Scrollbar

llm = OpenAI(model_name="text-davinci-003", temperature=0)
tools = load_tools(['serpapi', 'requests_all'], llm=llm, serpapi_api_key=serp_api_key)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

## Function to handle the query button click
def run_query():
    query_text = query_entry.get()
    result = agent.run(query_text)
    result_text.config(state=tk.NORMAL)  # Enable the Text widget for editing
    result_text.delete("1.0", tk.END)  # Clear previous text
    result_text.insert(tk.END, textwrap.fill(result, width=50))  # Insert the result
    result_text.config(state=tk.DISABLED)  # Disable the Text widget for editing

# Create the main application window
app = tk.Tk()
app.title("AI Query Agent")

# Create and pack GUI components
query_label = Label(app, text="Enter your query:")
query_label.pack(pady=10)

query_entry = Entry(app, width=40)
query_entry.pack()

query_button = Button(app, text="Run Query", command=run_query)
query_button.pack(pady=10)

result_text = Text(app, wrap=tk.WORD, width=50, height=10)
result_text.pack(padx=20, pady=10)

# Add a scrollbar for the result_text Text widget
scrollbar = Scrollbar(app, command=result_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
result_text.config(yscrollcommand=scrollbar.set)

# Disable the result_text widget to make it read-only
result_text.config(state=tk.DISABLED)

# Start the Tkinter main loop
app.mainloop()

query_button = Button(app, text="Run Query", command=run_query)
query_button.pack(pady=10)

result_text = Label(app, text="", wraplength=400, justify="left")
result_text.pack(padx=20, pady=10)

# Start the Tkinter main loop
app.mainloop()
