"""The Wikipedia API tool in LangChain is a powerful tool that allows language models
to interact with the Wikipedia API to fetch information and use it to answer questions. """

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType
import textwrap

app = Flask(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_key = os.environ.get("GOOGLE_CSE_ID")

# Initialize OpenAI and tools
llm = OpenAI(model_name="text-davinci-003", temperature=0)
tools = load_tools(["wikipedia"])
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Define a route to render the HTML template


# Define a route to handle user input and generate replies
@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""

    if request.method == "POST":
        query = request.form.get("query")
        reply = agent.run(query)
        #reply = textwrap.fill(reply, width=60)  # Wrap the reply for better display

    return render_template("index.html", reply=reply)

if __name__ == '__main__':
    app.run(debug=True)
