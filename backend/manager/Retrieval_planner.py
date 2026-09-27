#!/usr/bin/env python3

"""
planner.py

Standalone planner for Jarvis.

The planner sends the user's query to the configured Qwen 3.6 model and expects a JSON
execution plan in return.

Requirements:
    pip install requests

Assumes llama-server is running on:

    http://localhost:8081
"""

import json
import requests
import re
from pathlib import Path

CONFIG_FILE = Path(__file__).with_name("config.json")
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    config = json.load(config_file)

llama_config = config["llama"]
planner_config = config["planner"]
LLAMA_SERVER = (
    f"http://{llama_config['client_host']}:{llama_config['port']}"
    "/v1/chat/completions"
)

PLANNER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "need_reasoning": {"type": "boolean"},
        "need_diary": {"type": "boolean"},
        "need_web": {"type": "boolean"},
        "diary_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3,
        },
        "web_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3,
        },
    },
    "required": [
        "need_reasoning",
        "need_diary",
        "need_web",
        "diary_queries",
        "web_queries",
    ],
    "additionalProperties": False,
}

def planner(user_query: str, prompt: str) -> dict:
    """
    Call the configured local model and return the planner JSON.
    """

    payload = {
        "model": planner_config["model"],
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        "temperature": planner_config["temperature"],
        "stream": False,

        "chat_template_kwargs": {
          "enable_thinking": False
        },

        "response_format": {
            "type": "json_schema",
            "json_schema": PLANNER_RESPONSE_SCHEMA,
        },
    }

    response = requests.post(
        LLAMA_SERVER,
        json=payload,
        timeout=planner_config["timeout_seconds"],
    )

    response.raise_for_status()

    resp = response.json()

    #print(json.dumps(resp, indent=2))

    content = resp["choices"][0]["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise RuntimeError(f"No JSON found:\n{content}")

    try:
      return json.loads(match.group(0))
    except json.JSONDecodeError:
      print("Planner returned invalid JSON:")
      print(match.group(0))

      return {
        "need_diary": False,
        "need_web": False,
        "need_reasoning": False,
        "diary_queries": [],
        "web_queries": [],
      }


def main():

    while True:

        query = input("\nUser > ")

        if query.lower() in {"exit", "quit"}:
            break

        try:

            result = planner(query)

            print("\nPlanner output:\n")
            print(json.dumps(result, indent=4))

        except Exception as e:

            print("\nPlanner failed:\n")
            print(e)


if __name__ == "__main__":
    main()
