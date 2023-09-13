import textwrap

with open('text.txt') as f:
    text = f.read()
wrapped_text = textwrap.fill(text, width=100)
print(wrapped_text)