import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

class QueryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Query Application")

        self.org_id_label = ttk.Label(root, text="Organization ID:")
        self.org_id_entry = ttk.Entry(root)
        self.dataset_label = ttk.Label(root, text="Dataset Name:")
        self.dataset_entry = ttk.Entry(root)
        self.question_label = ttk.Label(root, text="Enter your question:")
        self.question_entry = ttk.Entry(root)
        self.query_button = ttk.Button(root, text="Query Dataset", command=self.query_dataset)
        self.answer_label = ttk.Label(root, text="Answer:")

        self.org_id_label.pack()
        self.org_id_entry.pack()
        self.dataset_label.pack()
        self.dataset_entry.pack()
        self.question_label.pack()
        self.question_entry.pack()
        self.query_button.pack()
        self.answer_label.pack()

    def query_dataset(self):
        org_id = self.org_id_entry.get()
        dataset_name = self.dataset_entry.get()
        question = self.question_entry.get()

        if not org_id or not dataset_name or not question:
            messagebox.showerror("Error", "Please provide valid Organization ID, Dataset Name, and a question.")
            return

        try:
            answer = self.run_query(org_id, dataset_name, question)
            self.answer_label.config(text=f"Answer: {answer}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def run_query(self, org_id: str, dataset_name: str, question: str):
        # TODO: Replace with your API keys
        embeddings = OpenAIEmbeddings()

        # TODO: Replace with your organization ID and dataset name
        dataset_path = f"hub://{org_id}/{dataset_name}"

        db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever())

        result = qa.run(question)
        return result

if __name__ == "__main__":
    root = tk.Tk()
    app = QueryApp(root)
    root.mainloop()
