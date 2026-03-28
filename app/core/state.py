import os

STATE_FILE = "data/uploads/.last_document"

def set_document(path: str):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(path)

def get_document() -> str | None:
    try:
        with open(STATE_FILE, "r") as f:
            path = f.read().strip()
            return path if os.path.exists(path) else None
    except:
        return None