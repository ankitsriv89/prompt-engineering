
from dotenv import load_dotenv
import os
load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from typing import List
import requests
import textwrap
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter

from langchain.document_loaders import PagedPDFSplitter

import tqdm
import os

class ReportUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Report Uploader")

        self.urls_label = ttk.Label(root, text="Enter URLs (separated by newline):")
        self.urls_text = scrolledtext.ScrolledText(root, width=40, height=10)
        self.org_id_label = ttk.Label(root, text="Organization ID:")
        self.org_id_entry = ttk.Entry(root)
        self.dataset_label = ttk.Label(root, text="Dataset Name:")
        self.dataset_entry = ttk.Entry(root)
        self.upload_button = ttk.Button(root, text="Upload Reports", command=self.upload_reports)

        self.urls_label.pack()
        self.urls_text.pack()
        self.org_id_label.pack()
        self.org_id_entry.pack()
        self.dataset_label.pack()
        self.dataset_entry.pack()
        self.upload_button.pack()

    def upload_reports(self):
        urls_input = self.urls_text.get("1.0", tk.END).strip().split('\n')
        org_id = self.org_id_entry.get()
        dataset_name = self.dataset_entry.get()

        if not urls_input or not org_id or not dataset_name:
            messagebox.showerror("Error", "Please provide valid URLs, Organization ID, and Dataset Name.")
            return

        try:
            pages = self.load_reports_from_urls(urls_input)
            self.create_dataset(pages, org_id, dataset_name)
            messagebox.showinfo("Success", "Reports uploaded and dataset created successfully.")
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

    def create_dataset(self, pages: List[str], org_id: str, dataset_name: str):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()

        # TODO: Replace with your organization ID and dataset name
        dataset_path = f"hub://{org_id}/{dataset_name}"

        db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
        db.add_documents(texts)

if __name__ == "__main__":
    root = tk.Tk()
    app = ReportUploaderApp(root)
    root.mainloop()
