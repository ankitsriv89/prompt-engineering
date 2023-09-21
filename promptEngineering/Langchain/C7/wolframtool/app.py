from dotenv import load_dotenv
import os

load_dotenv()
# Access the environment variables from the .env file
openai_api_key = os.environ.get('OPENAI_API_KEY')
wolfram_alpha_appid = os.environ.get('WOLFRAM_ALPHA_APPID')

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from langchain. utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)
#wolfram = WolframAlphaAPIWrapper()
#result = wolfram.run("What is integral of x^2 + y^2")
#print(result)  # Output: 'x = 2/5'

tools = load_tools(["wolfram-alpha", "wikipedia"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""

    if request.method == "POST":
        query = request.form.get("query")
        reply = agent.run(query)

    return render_template("index.html", reply=reply)

if __name__ == "__main__":
    app.run(debug=True)