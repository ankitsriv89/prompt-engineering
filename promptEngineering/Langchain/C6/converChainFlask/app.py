from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import OpenAI, ConversationChain

app = Flask(__name__)

llm = OpenAI(model_name="text-davinci-003", temperature=0)

memory = ConversationBufferMemory(return_messages=True)

@app.route('/')
def chatbot():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    input_text = request.form['input_text']

    if input_text.lower() == 'bye':
        return jsonify({'response': 'Goodbye!'})

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    response = conversation.predict(input=input_text)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
