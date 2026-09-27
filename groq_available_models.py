import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("GROQ_FREE_API_KEY")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}",
    },
)

print(response.status_code)
print(json.dumps(response.json(), indent=4))