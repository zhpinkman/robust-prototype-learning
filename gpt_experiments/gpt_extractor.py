import requests
import json
from tqdm import tqdm
import time
import argparse
import os


API_KEY = os.environ["OPENAI_API_KEY"]
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def generate_chat_completion(
    system_content,
    user_content,
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=1,
    top_p=0.9,
):
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    response = requests.post(
        API_ENDPOINT, headers=headers, data=json.dumps(data), timeout=60
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
