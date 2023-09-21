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

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button

llm = OpenAI(model_name="text-davinci-003", temperature=0)

memory = ConversationBufferMemory(return_messages=True)

class ChatbotApp(App):
    def build(self):
        self.title = 'AI Chatbot'
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.query_label = Label(text="Enter your query:")
        self.query_entry = TextInput()
        self.query_entry.bind(on_text_validate=self.on_submit)

        self.response_text = TextInput(hint_text="AI Responses", readonly=True, multiline=True)

        self.submit_button = Button(text="Submit")
        self.submit_button.bind(on_press=self.on_submit)

        self.layout.add_widget(self.query_label)
        self.layout.add_widget(self.query_entry)
        self.layout.add_widget(self.response_text)
        self.layout.add_widget(self.submit_button)

        return self.layout

    def on_submit(self, instance):
        input_text = self.query_entry.text
        if input_text.lower() == 'bye':
            App.get_running_app().stop()
            return

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
        response = conversation.predict(input=input_text)

        self.response_text.text += f"User: {input_text}\n"
        self.response_text.text += f"AI: {response}\n\n"
        self.query_entry.text = ""

if __name__ == '__main__':
    ChatbotApp().run()
