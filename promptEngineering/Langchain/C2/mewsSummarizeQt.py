import sys
import os
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from newspaper import Article
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

class ArticleSummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Article Summarizer")
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: lightblue;")

        self.label_url = QLabel("Enter Article URL:", self)
        self.label_url.setGeometry(20, 20, 150, 30)

        self.entry_url = QLineEdit(self)
        self.entry_url.setGeometry(170, 20, 200, 30)

        self.summarize_button = QPushButton("Summarize Article", self)
        self.summarize_button.setGeometry(150, 70, 150, 30)
        self.summarize_button.clicked.connect(self.start_summarization)

        self.summarization_thread = None  # Thread for article summarization

    def start_summarization(self):
        if self.summarization_thread is None or not self.summarization_thread.isRunning():
            self.summarization_thread = SummarizationThread(self.entry_url.text())
            self.summarization_thread.summary_ready.connect(self.on_summarization_finished)
            self.summarize_button.setEnabled(False)  # Disable button during processing
            self.summarization_thread.start()

    def on_summarization_finished(self, summary):
        QMessageBox.information(self, "Summary", summary)
        self.summarize_button.setEnabled(True)  # Enable button after processing

class SummarizationThread(QThread):
    summary_ready = pyqtSignal(str)

    def __init__(self, article_url):
        super().__init__()
        self.article_url = article_url

    def run(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        }
        session = requests.Session()

        try:
            response = session.get(self.article_url, headers=headers, timeout=10)

            if response.status_code == 200:
                article = Article(self.article_url)
                article.download()
                article.parse()

                article_title = article.title
                article_text = article.text

                template = f"""You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""
                prompt = template.format(article_title=article.title, article_text=article.text)

                messages = [HumanMessage(content=prompt)]
                chat = ChatOpenAI(model_name="gpt-4", temperature=0)
                summary = chat(messages)

                self.summary_ready.emit(summary.content)
            else:
                self.summary_ready.emit(f"Failed to fetch article at {self.article_url}")

        except Exception as e:
            self.summary_ready.emit(f"Error occurred while fetching article at {self.article_url}: {e}")

def main():
    app = QApplication(sys.argv)
    window = ArticleSummarizerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
