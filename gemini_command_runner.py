# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv


from command import execute_command

def get_command_from_gemini(user_input):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    client = genai.Client(
        api_key=api_key,
    )
    model = "gemini-2.5-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="Respond ONLY in JSON format as: { \"command\": \"<windows_command>\" }. For the given user input, provide the Windows command prompt command in the 'command' field. Do not include any explanation or extra text."),
        ],
    )
    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text
    try:
        command_json = json.loads(response)
        return command_json.get("command", "")
    except Exception as e:
        print("Error parsing Gemini response as JSON:", e)
        print("Raw response:", response)
        return ""

if __name__ == "__main__":
    user_input = input("Enter your request: ")
    command = get_command_from_gemini(user_input)
    print(f"Gemini generated command: {command}")
    output = execute_command(command)
    print("\nCommand Output:\n", output)
