# this code works on older version of openai paackage that is 0.28.0. As of now we have installed the newer version.

import openai
import os

# Ensure the API key is set correctly
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is not set. Please set the OPENAI_API_KEY environment variable.")

openai.api_key = api_key

try:
    # List available models using the new API interface
    models = openai.Model.list()
    model_ids = [model.id for model in models.data]
    print("Available models:", model_ids)
except openai.OpenAIError as e:
    print(f"Error: {e}")
