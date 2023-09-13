import tkinter as tk
from tkinter import filedialog, scrolledtext
from dotenv import load_dotenv
import os
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader

class SummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Summarizer")
        self.root.geometry("600x400")

        self.load_dotenv()
        self.create_widgets()

    def load_dotenv(self):
        load_dotenv()
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')

    def create_widgets(self):
        self.path_label = tk.Label(self.root, text="File Path:")
        self.path_label.pack(pady=10)

        self.path_entry = tk.Entry(self.root, width=40)
        self.path_entry.pack()

        self.load_button = tk.Button(self.root, text="Load Document", command=self.load_document)
        self.load_button.pack(pady=10)

        self.summarize_button = tk.Button(self.root, text="Summarize", command=self.summarize_document)
        self.summarize_button.pack()

        self.summary_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=60, height=10)
        self.summary_text.pack(pady=10)

    def load_document(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")])
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)

    def summarize_document(self):
        file_path = self.path_entry.get()
        if not file_path:
            return

        llm = OpenAI(model_name="text-davinci-003", temperature=0)
        summarize_chain = load_summarize_chain(llm)

        document_loader = PyPDFLoader(file_path=file_path)
        document = document_loader.load()

        summary = summarize_chain(document)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary['output_text'])

def main():
    root = tk.Tk()
    app = SummarizerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
