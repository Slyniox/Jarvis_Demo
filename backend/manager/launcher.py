#!/usr/bin/env python3

import json
from pathlib import Path

from model_manager import ModelManager

CONFIG_FILE = Path(__file__).parent / "config.json"


def load_config():

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


config = load_config()

manager = ModelManager(config["llama"])
