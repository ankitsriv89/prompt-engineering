import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames
from typing import List
import requests

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
from langchain.document_loaders import PagedPDFSplitter
from langchain.chat_models import ChatOpenAI
import tqdm
import os


class FinancialReportLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Report Loader")

        self.urls_label = ttk.Label(root, text="Enter URLs (separated by newline):")
        self.urls_text = scrolledtext.ScrolledText(root, width=40, height=10)
        self.org_id_label = ttk.Label(root, text="Organization ID:")
        self.org_id_entry = ttk.Entry(root)
        self.dataset_label = ttk.Label(root, text="Dataset Name:")
        self.dataset_entry = ttk.Entry(root)
        self.question_label = ttk.Label(root, text="Enter your question:")
        self.question_entry = ttk.Entry(root)
        self.load_button = ttk.Button(root, text="Load Reports", command=self.load_reports)

        self.urls_label.pack()
        self.urls_text.pack()
        self.org_id_label.pack()
        self.org_id_entry.pack()
        self.dataset_label.pack()
        self.dataset_entry.pack()
        self.question_label.pack()
        self.question_entry.pack()
        self.load_button.pack()

    def load_reports(self):
        urls_input = self.urls_text.get("1.0", tk.END).strip().split('\n')
        org_id = self.org_id_entry.get()
        dataset_name = self.dataset_entry.get()
        question = self.question_entry.get()

        if not urls_input or not org_id or not dataset_name or not question:
            messagebox.showerror("Error", "Please provide valid URLs, Organization ID, Dataset Name, and a question.")
            return

        try:
            pages = self.load_reports_from_urls(urls_input)
            self.create_dataset(pages, org_id, dataset_name, question)
            messagebox.showinfo("Success", "Reports loaded and dataset created successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def load_reports_from_urls(self, urls: List[str]) -> List[str]:
        pages = []

        for url in tqdm.tqdm(urls):
            r = requests.get(url)
            path = url.split('/')[-1]
            with open(path, 'wb') as f:
                f.write(r.content)
            loader = PagedPDFSplitter(path)
            local_pages = loader.load_and_split()
            pages.extend(local_pages)
            os.remove(path)  # Remove downloaded PDF after processing
        return pages

    def create_dataset(self, pages: List[str], org_id: str, dataset_name: str, question: str):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()

        # TODO: Replace with your organization ID and dataset name
        dataset_path = f"hub://{org_id}/{dataset_name}"

        db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
        db.add_documents(texts)

        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever())

        result = qa.run(question)
        messagebox.showinfo("Answer", f"Answer: {result}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialReportLoaderApp(root)
    root.mainloop()
