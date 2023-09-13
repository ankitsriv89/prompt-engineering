import tkinter as tk
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import PromptTemplate, OpenAI
from typing import Dict, List

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}


def get_concatenated_output():
    word = word_entry.get()

    llm = OpenAI(model_name="text-davinci-003", temperature=0)

    prompt_1 = PromptTemplate(
        input_variables=["word"],
        template="What is the meaning of the following word '{word}'?",
    )
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables=["word"],
        template="provide 5 words to replace the following: {word}?",
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
    concat_output = concat_chain.run(word)

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, concat_output)


app = tk.Tk()
app.title("word App")

frame = tk.Frame(app, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Enter a word:")
label.grid(row=0, column=0)

word_entry = tk.Entry(frame)
word_entry.grid(row=0, column=1)

concatenate_button = tk.Button(frame, text="Show", command=get_concatenated_output)
concatenate_button.grid(row=0, column=2)

result_text = tk.Text(frame, height=10, width=40)
result_text.grid(row=1, column=0, columnspan=3, pady=10)

app.mainloop()
