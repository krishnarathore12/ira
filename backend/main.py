import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load your .env file
load_dotenv()

# 2. Initialize the client
# It automatically looks for the OPENAI_API_KEY environment variable
client = OpenAI()

# 3. Initialize conversation history
# The 'system' message sets the persona/behavior of the assistant
messages = [
    {"role": "system", "content": "You are a helpful and concise assistant."}
]

print("--- AI Chat Started (Type 'quit' to stop) ---")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit", "bye"]:
        break

    # Add the user's message to history
    messages.append({"role": "user", "content": user_input})

    # 4. Request a completion from the model
    # Using 'gpt-4o' or 'gpt-3.5-turbo' for standard chat
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    # 5. Extract and print the assistant's reply
    assistant_reply = response.choices[0].message.content
    print(f"AI: {assistant_reply}")

    # Add the assistant's reply back to history to maintain context
    messages.append({"role": "assistant", "content": assistant_reply})
