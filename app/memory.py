import json
import os
from datetime import datetime
import uuid

MEMORY_FILE = "chats.json"


def load_chats():
    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_chats(chats):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chats, f, indent=2)


def create_new_chat(name="New Chat"):
    return {
        'id': str(uuid.uuid4()),
        'name': name,
        'date': datetime.now().isoformat(),
        'messages': []
    }