import json
import os
from datetime import datetime

REGISTRY_DIR = "models_registry"


def ensure_registry_dir():
    if not os.path.exists(REGISTRY_DIR):
        os.makedirs(REGISTRY_DIR)


def load_registry():
    ensure_registry_dir()
    models = []
    for filename in os.listdir(REGISTRY_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(REGISTRY_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["id"] = filename.replace(".json", "")
                    models.append(data)
            except Exception:
                pass
    return models


def register_model(model_info):
    ensure_registry_dir()
    model_id = model_info.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_id}.json"
    filepath = os.path.join(REGISTRY_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    return model_id