import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv
import os
import json
import requests
from newspaper import Article
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the Tkinter application
app = tk.Tk()
app.title("Article Summarizer")

# Create a function to fetch and summarize the article
def summarize_article():
    article_url = entry_url.get()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()

    try:
        response = session.get(article_url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()

            article_title = article.title
            article_text = article.text

            # Prepare template for prompt
            template = f"""You are a very good assistant that summarizes online articles.
                            Here's the article you want to summarize.
                            ==================
                            Title: {article_title}
                                    {article_text}
                            ==================

                            Write a summary of the previous article.
                        """
            prompt = template.format(article_title=article.title, article_text=article.text)

            # Generate summary
            messages = [HumanMessage(content=prompt)]
            chat = ChatOpenAI(model_name="gpt-4", temperature=0)
            summary = chat(messages)

            # Display the summary in a messagebox
            messagebox.showinfo("Summary", summary.content)

        else:
            messagebox.showerror("Error", f"Failed to fetch article at {article_url}")

    except Exception as e:
        messagebox.showerror("Error", f"Error occurred while fetching article at {article_url}: {e}")

# Create GUI elements
label_url = tk.Label(app, text="Enter Article URL:")
entry_url = tk.Entry(app, width=50)
summarize_button = tk.Button(app, text="Summarize Article", command=summarize_article)

# Place GUI elements in the window
label_url.pack()
entry_url.pack()
summarize_button.pack()

# Start the Tkinter main loop
app.mainloop()
