

import os

from openai import OpenAI

from config import PROMPT_CONFIG

api_key = PROMPT_CONFIG.get("api_key") or os.getenv(PROMPT_CONFIG.get("api_key_env", "OPENAI_API_KEY"), "")
client_kwargs = {"api_key": api_key}
base_url = PROMPT_CONFIG.get("base_url")
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)
MODEL_NAME = PROMPT_CONFIG.get("model", "gpt-3.5-turbo")


def openai_api(question):
    sign = True
    while sign:
        try:
            rsp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": question},
                ],
                timeout=60.0,
            )
            sign = False
        except Exception as e:
            print(f"[WARN] API call failed, retrying... error: {e}")
            sign = True
    return rsp.choices[0].message.content
