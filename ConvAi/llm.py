from openai import OpenAI
from prompt import system_instruction

client = OpenAI()

messages = [
    {"role" : "system", "content" : system_instruction}	
]

def chat(messages, model="gpt-3.5-turbo", temperature=0.9):
    response = client.chat.completions.create(
        model=model,
        messages= messages,
        temperature=temperature,
    )

    return response.choices[0].message.content