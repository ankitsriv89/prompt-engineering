""" use a few shot prompts to teach the LLM by providing examples to respond sarcastically to questions."""

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Create the LLMChain for the few_shot_prompt_template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "How can I take a woman to bed?"}
response = chain.run(input_data)

print(response)

"""dynamic prompts: Instead of using a static template, this approach incorporates examples of previous 
interactions, allowing the AI better to understand the context and style of the desired response.

Dynamic prompts offer several advantages over static templates:
Improved context understanding: By providing examples, the AI can grasp the context and style of responses 
more effectively, enabling it to generate responses that are more in line with the desired output.

Flexibility: Dynamic prompts can be easily customized and adapted to specific use cases, allowing developers 
to experiment with different prompt structures and find the most effective format for their application.
    
Better results: As a result of the improved context understanding and flexibility,
dynamic prompts often yield higher-quality outputs that better match user expectations.
"""