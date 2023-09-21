import tkinter as tk
from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import OpenAI, ConversationChain

llm = OpenAI(model_name="text-davinci-003", temperature=0)

memory = ConversationBufferMemory(return_messages=True)

def submit_query():
    input_text = query_entry.get()
    if input_text.lower() == 'bye':
        root.destroy()  # Close the GUI when the user enters 'no'.
        return

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    response = conversation.predict(input=input_text)

    response_text.config(state=tk.NORMAL)
    response_text.insert(tk.END, f"User: {input_text}\n")
    response_text.insert(tk.END, f"AI: {response}\n\n")
    response_text.config(state=tk.DISABLED)
    query_entry.delete(0, tk.END)  # Clear the input field

# Create the main GUI window
root = tk.Tk()
root.title("AI Chatbot")

# Create input field
query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=10)
query_entry.insert(0, "Enter your query (type 'no' to exit)")

# Create response text area
response_text = tk.Text(root, width=50, height=20, state=tk.DISABLED)
response_text.pack()

# Create submit button
submit_button = tk.Button(root, text="Submit", command=submit_query)
submit_button.pack()

root.mainloop()
