import tkinter as tk
from tkinter import Label, Entry, Button, Text
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
huggingfacehub_api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Initialize ChatOpenAI and LLMChain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"],
                                    template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)


# Function to handle translation
def translate_text():
    source_language = source_language_entry.get()
    target_language = target_language_entry.get()
    text = text_to_translate_entry.get()

    translated_text = translation_chain.predict(source_language=source_language, target_language=target_language,
                                                text=text)

    translated_text_box.delete(1.0, tk.END)  # Clear the text box
    translated_text_box.insert(tk.END, translated_text)


# Create the main application window
app = tk.Tk()
app.title("Language Translation")

# Labels and Entry Fields
source_language_label = Label(app, text="Source Language:")
source_language_label.pack()
source_language_entry = Entry(app)
source_language_entry.pack()

target_language_label = Label(app, text="Target Language:")
target_language_label.pack()
target_language_entry = Entry(app)
target_language_entry.pack()

text_to_translate_label = Label(app, text="Text to Translate:")
text_to_translate_label.pack()
text_to_translate_entry = Entry(app)
text_to_translate_entry.pack()

# Translate Button
translate_button = Button(app, text="Translate", command=translate_text)
translate_button.pack()

# Translated Text Box
translated_text_box = Text(app, height=10, width=40)
translated_text_box.pack()

app.mainloop()
