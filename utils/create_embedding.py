import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "text-embedding-3-small"

client = OpenAI(base_url=endpoint, api_key=token)


def create_embedding(text: str) -> List[float]:
    response = client.embeddings.create(model=model_name, input=text)
    return response.data[0].embedding
