from flask import Flask, render_template, request
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
tools = load_tools(["python_repl"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query_text = request.form.get("query")
        result = agent(query_text)
        return render_template("index.html", query=query_text, result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
