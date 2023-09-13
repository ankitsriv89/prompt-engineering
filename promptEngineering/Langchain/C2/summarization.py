import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, \
    QLabel, QPushButton, QFileDialog, QLineEdit, QPlainTextEdit
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from dotenv import load_dotenv
import os
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from langchain.document_loaders import PyPDFLoader

class SummarizerThread(QThread):
    summary_ready = pyqtSignal(str)

    def __init__(self, file_path, openai_api_key):
        super().__init__()
        self.file_path = file_path
        self.openai_api_key = openai_api_key

    def run(self):
        llm = OpenAI(model_name='text-davinci-003', temperature=0)
        summarize_chain = load_summarize_chain(llm)

        document_loader = PyPDFLoader(file_path=self.file_path)
        document = document_loader.load()

        summary = summarize_chain(document)
        self.summary_ready.emit(summary['output_text'])

class SummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Document Summarizer')
        self.setGeometry(100, 100, 600, 400)

        # Set background color
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(210, 240, 240))
        self.setPalette(palette)

        self.path_label = QLabel('File Path:')
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText('Enter file path here')
        self.load_button = QPushButton('Load Document', self)
        self.load_button.clicked.connect(self.load_document)
        self.summary_button = QPushButton('Summarize', self)
        self.summary_button.clicked.connect(self.summarize_document)
        self.summary_text = QPlainTextEdit(self)
        self.summary_text.setPlaceholderText('Summary will appear here')
        self.summary_text.setReadOnly(True)

        # Set widget background color
        widget_palette = QPalette()
        widget_palette.setColor(QPalette.Window, QColor(210, 225, 255))
        self.path_edit.setPalette(widget_palette)
        self.summary_text.setPalette(widget_palette)

        vbox = QVBoxLayout()
        vbox.addWidget(self.path_label)
        vbox.addWidget(self.path_edit)
        vbox.addWidget(self.load_button)
        vbox.addWidget(self.summary_button)
        vbox.addWidget(self.summary_text)

        self.setLayout(vbox)

    def load_document(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open PDF File', '', 'PDF Files (*.pdf);;All Files (*)', options=options)
        if file_path:
            self.path_edit.setText(file_path)

    def summarize_document(self):
        file_path = self.path_edit.text()
        if not file_path:
            return

        load_dotenv()
        openai_api_key = os.environ.get('OPENAI_API_KEY')

        self.summarizer_thread = SummarizerThread(file_path, openai_api_key)
        self.summarizer_thread.summary_ready.connect(self.display_summary)
        # Disable the "Summarize" button while summarization is in progress
        self.summary_button.setEnabled(False)
        self.summarizer_thread.start()

    def display_summary(self, summary):
        self.summary_text.setPlainText(summary)
        self.summary_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = SummarizerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
